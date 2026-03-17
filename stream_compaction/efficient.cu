#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include <vector>
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        #define blockSize 256
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
        * Work-efficient (Blelloch) scan: up-sweep and down-sweep kernels.
        *
        * We operate on an array sized to the next power-of-two >= n.
        * Up-sweep: for each stride = 2^(d+1), threads reduce pairs into the right element.
        * Down-sweep: for each stride (descending), threads swap and propagate partial sums.
        */

        __global__ void kernUpSweep(int nPow2, int stride, int* data) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            int idx = tid * stride + stride - 1;
            if (idx >= nPow2) return;
            int left = idx - (stride >> 1);
            data[idx] += data[left];
        }

        __global__ void kernDownSweep(int nPow2, int stride, int* data) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            int idx = tid * stride + stride - 1;
            if (idx >= nPow2) return;
            int left = idx - (stride >> 1);
            int t = data[left];
            data[left] = data[idx];
            data[idx] += t;
        }

        /*
         * Per-block Blelloch (shared-memory) exclusive scan kernel.
         *
         * - Each block loads up to blockDim.x elements into shared memory (pad with 0).
         * - Performs up-sweep to build partial sums, captures block sum to d_blockSums,
         *   sets last element to 0, then performs down-sweep to produce an exclusive
         *   scan of the block in shared memory.
         * - Writes per-element exclusive scan results to d_odata.
         */
        __global__ void kernBlockScan(int n, const int* d_idata, int* d_odata, int* d_blockSums) {
            __shared__ int s_data[blockSize];

            const int blockStart = blockIdx.x * blockDim.x;
            const int thid = threadIdx.x;
            const int gid = blockStart + thid;

            // load (pad out-of-range with 0)
            s_data[thid] = (gid < n) ? d_idata[gid] : 0;
            __syncthreads();

            // up-sweep (reduce)
            for (int stride = 1; stride < blockDim.x; stride <<= 1) {
                int idx = (thid + 1) * 2 * stride - 1;
                if (idx < blockDim.x) {
                    s_data[idx] += s_data[idx - stride];
                }
                __syncthreads();
            }

            // save total sum for this block
            if (thid == 0) {
                d_blockSums[blockIdx.x] = s_data[blockDim.x - 1];
                s_data[blockDim.x - 1] = 0; // prepare for exclusive scan
            }
            __syncthreads();

            // down-sweep
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                int idx = (thid + 1) * 2 * stride - 1;
                if (idx < blockDim.x) {
                    int t = s_data[idx - stride];
                    s_data[idx - stride] = s_data[idx];
                    s_data[idx] += t;
                }
                __syncthreads();
            }

            // write result back
            if (gid < n) {
                d_odata[gid] = s_data[thid];
            }
        }

        // Add scanned block offsets to each element in the block
        __global__ void kernAddBlockOffsets(int n, int* d_data, const int* d_blockOffsets) {
            int gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid >= n) return;
            int add = d_blockOffsets[blockIdx.x];
            d_data[gid] += add;
        }

        /**
         * Ultra per-block counting using warp-level primitives.
         *
         * Produces one count per block. This avoids nondeterministic global atomics
         * so we can keep the compaction stable by scanning block counts.
         */
        __global__ void kernCountUltra(int n, const int* d_input, int* d_blockCounts) {
            const int gid = blockIdx.x * blockDim.x + threadIdx.x;
            const int lane = threadIdx.x & 31;
            const int warpId = threadIdx.x >> 5;
            const int numWarps = blockDim.x >> 5;

            __shared__ int s_warpCounts[blockSize / 32];

            const bool pred = (gid < n) && (d_input[gid] != 0);

            const unsigned int warpMask = __ballot_sync(0xffffffffu, pred);
            const int warpCount = __popc(warpMask);

            if (lane == 0) {
                s_warpCounts[warpId] = warpCount;
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                int running = 0;
                for (int w = 0; w < numWarps; ++w) {
                    running += s_warpCounts[w];
                }
                d_blockCounts[blockIdx.x] = running;
            }
        }

        /**
         * Ultra per-block scatter using warp-level offsets plus precomputed block bases.
         *
         * This is stable because d_blockOffsets is an exclusive scan of block counts.
         */
        __global__ void kernScatterUltra(int n, const int* d_input, int* d_output, const int* d_blockOffsets) {
            const int gid = blockIdx.x * blockDim.x + threadIdx.x;
            const int lane = threadIdx.x & 31;
            const int warpId = threadIdx.x >> 5;
            const int numWarps = blockDim.x >> 5;

            __shared__ int s_warpCounts[blockSize / 32];
            __shared__ int s_warpOffsets[blockSize / 32];

            const bool pred = (gid < n) && (d_input[gid] != 0);

            const unsigned int warpMask = __ballot_sync(0xffffffffu, pred);
            const int warpPrefix = __popc(warpMask & ((1u << lane) - 1));
            const int warpCount = __popc(warpMask);

            if (lane == 0) {
                s_warpCounts[warpId] = warpCount;
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                int running = 0;
                for (int w = 0; w < numWarps; ++w) {
                    s_warpOffsets[w] = running;
                    running += s_warpCounts[w];
                }
            }
            __syncthreads();

            if (pred) {
                const int blockBase = d_blockOffsets[blockIdx.x];
                const int outIdx = blockBase + s_warpOffsets[warpId] + warpPrefix;
                d_output[outIdx] = d_input[gid];
            }
        }

        void scan(int n, int *odata, const int *idata) {
            // TODO
            // next power-of-two size
            int log2ceil = ilog2ceil(n);
            int nPow2 = 1 << log2ceil;

            int* d_worker; 
            cudaMalloc((void**)&d_worker, nPow2 * sizeof(int));

            // copy input from host to device (Host -> Device)
            cudaMemcpy(d_worker, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (nPow2 > n) {
                // zero the tail
                cudaMemset(d_worker + n, 0, (nPow2 - n) * sizeof(int));
            }

            timer().startGpuTimer();
            // Up-sweep (reduce) phase
            for (int d = 0; d < log2ceil; ++d) {
                int stride = 1 << (d + 1); // 2,4,8,...
                int active = nPow2 / stride; // number of operations
                dim3 blocks((active + blockSize - 1) / blockSize);
                kernUpSweep << <blocks, blockSize >> > (nPow2, stride, d_worker);
                cudaDeviceSynchronize();
                checkCUDAError("kernUpSweep");
            }

            // Set last element to zero for exclusive scan
            cudaMemset(d_worker + (nPow2 - 1), 0, sizeof(int));

            // Down-sweep phase
            for (int d = log2ceil - 1; d >= 0; --d) {
                int stride = 1 << (d + 1);
                int active = nPow2 / stride;
                dim3 blocks((active + blockSize - 1) / blockSize);
                kernDownSweep << <blocks, blockSize >> > (nPow2, stride, d_worker);
                cudaDeviceSynchronize();
                checkCUDAError("kernDownSweep");
            }
            timer().endGpuTimer();

            // copy result back from device to host (Device -> Host)
            cudaMemcpy(odata, d_worker, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(d_worker);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return 0;
            }

            const int numBlocks = (n + blockSize - 1) / blockSize;

            int* d_bool = nullptr;
            int* d_indice = nullptr;
            int* d_input = nullptr;
            int* d_output = nullptr;
            int* d_blockSums = nullptr;
            int* d_blockOffsets = nullptr;

            cudaMalloc((void**)&d_bool, n * sizeof(int));
            cudaMalloc((void**)&d_indice, n * sizeof(int));
            cudaMalloc((void**)&d_input, n * sizeof(int));
            cudaMalloc((void**)&d_output, n * sizeof(int));
            cudaMalloc((void**)&d_blockSums, numBlocks * sizeof(int));
            cudaMalloc((void**)&d_blockOffsets, numBlocks * sizeof(int));
            checkCUDAError("cudaMalloc (efficient compact)");

            cudaMemcpy(d_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy input (efficient compact)");

            dim3 blocks(numBlocks);
            dim3 threads(blockSize);

            timer().startGpuTimer();

            // Phase 1: map to boolean mask
            StreamCompaction::Common::kernMapToBoolean<<<blocks, threads>>>(n, d_bool, d_input);
            cudaDeviceSynchronize();
            checkCUDAError("kernMapToBoolean (efficient compact)");

            // Phase 2: per-block shared-memory exclusive scan over the mask
            kernBlockScan<<<blocks, threads>>>(n, d_bool, d_indice, d_blockSums);
            cudaDeviceSynchronize();
            checkCUDAError("kernBlockScan (efficient compact)");

            // Phase 3: exclusive scan of block sums on host, then upload offsets
            std::vector<int> h_blockSums(numBlocks);
            std::vector<int> h_blockOffsets(numBlocks);
            cudaMemcpy(h_blockSums.data(), d_blockSums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy block sums to host (efficient compact)");

            int running = 0;
            for (int i = 0; i < numBlocks; ++i) {
                h_blockOffsets[i] = running;
                running += h_blockSums[i];
            }

            cudaMemcpy(d_blockOffsets, h_blockOffsets.data(), numBlocks * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy block offsets to device (efficient compact)");

            // Phase 4: add block offsets to per-element indices
            kernAddBlockOffsets<<<blocks, threads>>>(n, d_indice, d_blockOffsets);
            cudaDeviceSynchronize();
            checkCUDAError("kernAddBlockOffsets (efficient compact)");

            // Phase 5: scatter
            StreamCompaction::Common::kernScatter<<<blocks, threads>>>(n, d_output, d_input, d_bool, d_indice);
            cudaDeviceSynchronize();
            checkCUDAError("kernScatter (efficient compact)");

            // Compute compacted count = last index + last bool
            int lastIndex = 0;
            int lastBool = 0;
            cudaMemcpy(&lastIndex, d_indice + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, d_bool + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy count (efficient compact)");
            const int count = lastIndex + lastBool;

            timer().endGpuTimer();

            if (count > 0) {
                cudaMemcpy(odata, d_output, count * sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAError("cudaMemcpy output (efficient compact)");
            }

            cudaFree(d_bool);
            cudaFree(d_indice);
            cudaFree(d_input);
            cudaFree(d_output);
            cudaFree(d_blockSums);
            cudaFree(d_blockOffsets);

            return count;
        }

        

        int compact_ultra(int n, int* odata, const int* idata) {
            if (n <= 0) {
                return 0;
            }

            const int numBlocks = (n + blockSize - 1) / blockSize;
            dim3 blocks(numBlocks);
            dim3 threads(blockSize);

            int* d_input = nullptr;
            int* d_output = nullptr;
            int* d_blockCounts = nullptr;
            int* d_blockOffsets = nullptr;

            cudaMalloc((void**)&d_input, n * sizeof(int));
            cudaMalloc((void**)&d_output, n * sizeof(int));
            cudaMalloc((void**)&d_blockCounts, numBlocks * sizeof(int));
            cudaMalloc((void**)&d_blockOffsets, numBlocks * sizeof(int));
            checkCUDAError("cudaMalloc (compact_ultra)");

            cudaMemcpy(d_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy input (compact_ultra)");

            timer().startGpuTimer();

            // Phase 1: count surviving elements per block (warp-optimized)
            kernCountUltra<<<blocks, threads>>>(n, d_input, d_blockCounts);
            cudaDeviceSynchronize();
            checkCUDAError("kernCountUltra");

            // Phase 2: exclusive scan block counts on host to get stable block bases
            std::vector<int> h_blockCounts(numBlocks);
            std::vector<int> h_blockOffsets(numBlocks);
            cudaMemcpy(h_blockCounts.data(), d_blockCounts, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy block counts to host (compact_ultra)");

            int running = 0;
            for (int i = 0; i < numBlocks; ++i) {
                h_blockOffsets[i] = running;
                running += h_blockCounts[i];
            }
            const int count = running;

            cudaMemcpy(d_blockOffsets, h_blockOffsets.data(), numBlocks * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy block offsets to device (compact_ultra)");

            // Phase 3: scatter with warp-optimized intra-block offsets
            kernScatterUltra<<<blocks, threads>>>(n, d_input, d_output, d_blockOffsets);
            cudaDeviceSynchronize();
            checkCUDAError("kernScatterUltra");

            timer().endGpuTimer();

            if (count > 0) {
                cudaMemcpy(odata, d_output, count * sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAError("cudaMemcpy output (compact_ultra)");
            }

            cudaFree(d_input);
            cudaFree(d_output);
            cudaFree(d_blockCounts);
            cudaFree(d_blockOffsets);

            return count;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         *
         * This implementation performs a per-block shared-memory (Blelloch) scan,
         * collects block sums, does an exclusive scan of block sums on the host,
         * then adds the scanned block offsets back to each block.
         */
        void scan_shared(int n, int* odata, const int* idata) {
            if (n <= 0) {
                return;
            }

            const int numBlocks = (n + blockSize - 1) / blockSize;

            int* d_input = nullptr;
            int* d_output = nullptr;
            int* d_blockSums = nullptr;
            int* d_blockOffsets = nullptr;

            cudaMalloc((void**)&d_input, n * sizeof(int));
            cudaMalloc((void**)&d_output, n * sizeof(int));
            cudaMalloc((void**)&d_blockSums, numBlocks * sizeof(int));
            cudaMalloc((void**)&d_blockOffsets, numBlocks * sizeof(int));
            checkCUDAError("cudaMalloc (naive block-shared scan)");

            // copy input to device
            cudaMemcpy(d_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy host->device (naive block-shared scan)");

            dim3 blocks(numBlocks);
            dim3 threads(blockSize);

            timer().startGpuTimer();

            // Phase 1: per-block shared-memory exclusive scan, produce block sums
            kernBlockScan << <blocks, threads >> > (n, d_input, d_output, d_blockSums);
            cudaDeviceSynchronize();
            checkCUDAError("kernBlockScan");

            // Phase 2: scan block sums on host (safe because number of blocks is small)
            std::vector<int> h_blockSums(numBlocks);
            std::vector<int> h_blockOffsets(numBlocks);
            cudaMemcpy(h_blockSums.data(), d_blockSums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy block sums to host");

            int running = 0;
            for (int i = 0; i < numBlocks; ++i) {
                h_blockOffsets[i] = running;
                running += h_blockSums[i];
            }

            // copy scanned block offsets back to device
            cudaMemcpy(d_blockOffsets, h_blockOffsets.data(), numBlocks * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy block offsets to device");

            // Phase 3: add block offsets to each element
            kernAddBlockOffsets << <blocks, threads >> > (n, d_output, d_blockOffsets);
            cudaDeviceSynchronize();
            checkCUDAError("kernAddBlockOffsets");

            timer().endGpuTimer();

            // copy result back
            cudaMemcpy(odata, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy device->host (naive block-shared scan)");

            // cleanup
            cudaFree(d_input);
            cudaFree(d_output);
            cudaFree(d_blockSums);
            cudaFree(d_blockOffsets);
        }
    }
}
