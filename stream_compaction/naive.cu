#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <vector>

namespace StreamCompaction {
    namespace Naive {
#define blockSize 256  

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernNaiveScan(int N, int offset, const int* d_idata, int* d_odata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= N) return;

            if (index >= offset) {
                d_odata[index] = d_idata[index - offset] + d_idata[index];
            }
            else {
                // copy input for indices that don't participate this pass
                d_odata[index] = d_idata[index];
            }
        }

        // Convert an inclusive scan result to an exclusive scan:
        // out[0] = 0, out[i] = in[i-1] for i >= 1
        __global__ void kernInclusiveToExclusive(int N, const int* d_in, int* d_out) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= N) return;

            if (index == 0) {
                d_out[0] = 0;
            }
            else {
                d_out[index] = d_in[index - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {


            int* d_workerA = nullptr;
            int* d_workerB = nullptr;

            cudaMalloc((void**)&d_workerA, n * sizeof(int));
            cudaMalloc((void**)&d_workerB, n * sizeof(int));
            
            // copy input from host to device (Host -> Device)
            cudaMemcpy(d_workerA, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 blocks((n + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            // offset loop: 1,2,4,8,... < n
            for (int offset = 1; offset < n; offset <<= 1) {
                kernNaiveScan << <blocks, blockSize >> > (n, offset, d_workerA, d_workerB);
                // swap buffers for next iteration
                std::swap(d_workerA, d_workerB);
            }

            // Convert inclusive result in d_workerA to exclusive result in d_workerB
            kernInclusiveToExclusive << <blocks, blockSize >> > (n, d_workerA, d_workerB);
            // d_workerB now contains exclusive scan; make it the buffer to copy from
            std::swap(d_workerA, d_workerB);
            timer().endGpuTimer();

            // copy result back from device to host (Device -> Host)
            cudaMemcpy(odata, d_workerA, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(d_workerA);
            cudaFree(d_workerB);

        }
        
        /**
        * Performs a naive scan with shared memory
        */
        __global__ void kernBlockScan(int N, int* d_worker, int* d_blockSums) {
            __shared__ int s_data[blockSize];

            const int blockStart = blockIdx.x * blockDim.x; 
            const int thid = threadIdx.x; 
            const int index = blockStart + thid; 
            
            // load into shared memory (pading with 0)
            s_data[thid] = (index < N) ? d_worker[index] : 0;
            __syncthreads();

            // block level san
            // for (int stride = 1; stride < blockDim.x; stride <<= 1) {
            //     int idx = thid - stride; 
            //     if (idx >= 0) s_data[thid] = s_data[idx] + s_data[thid]; 
            //     __syncthreads(); 
            // }

            for (int stride = 1; stride < blockDim.x; stride <<= 1) {
                int val = s_data[thid];
                if (thid >= stride) {
                    val += s_data[thid - stride];
                }
                __syncthreads();       // ensure all reads are from prior state
                s_data[thid] = val;
                __syncthreads();
    }

            // save total sum for this block j
            if (thid == blockSize - 1) {
                d_blockSums[blockIdx.x] = s_data[blockDim.x - 1]; 
            }
            __syncthreads(); 

            // put data back 
            if (index < N) {
                d_worker[index] = s_data[thid];
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
        * Performs prefix-sum (aka scan) on idata using shared memory, 
        * storing the result into odata.
        */
        void scan_shared(int n, int* odata, const int* idata) {
            const int numBlocks = (n + blockSize - 1) / blockSize;
            int* d_worker = nullptr;
            int* d_output = nullptr; 
            int* d_blockSums = nullptr;
            int* d_blockOffsets = nullptr;

            cudaMalloc((void**)&d_worker, n * sizeof(int));
            cudaMalloc((void**)&d_output, n * sizeof(int));
            cudaMalloc((void**)&d_blockSums, numBlocks * sizeof(int));
            cudaMalloc((void**)&d_blockOffsets, numBlocks * sizeof(int));
            checkCUDAError("cudaMalloc (naive block-shared scan)");

            // copy input from host to device (Host -> Device)
            cudaMemcpy(d_worker, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy host->device (naive block-shared scan)");

            dim3 blocks(numBlocks);
            dim3 threads(blockSize);

            timer().startGpuTimer();

            // Phase 1: per-block shared-memory inclusive scan, produce block sums 
            kernBlockScan << <blocks, threads >> > (n, d_worker, d_blockSums);
            cudaDeviceSynchronize();
            checkCUDAError("kernBlockScan");

            // Phase 2: scan block sums on host (safe because number of blocks is small)
            std::vector<int> h_blockSums(numBlocks);
            std::vector<int> h_blockOffsets(numBlocks);
            cudaMemcpy(h_blockSums.data(), d_blockSums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost); 
            checkCUDAError("cudaMemcpy block sums to host");

            int running = 0;
            for (int i = 0; i < numBlocks; i++) {
                h_blockOffsets[i] = running; 
                running += h_blockSums[i];
            }
            // copy scanned block offsets back to device
            cudaMemcpy(d_blockOffsets, h_blockOffsets.data(), numBlocks * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy block offsets to device");

            // Phase 3: add block offsets to each element
            kernAddBlockOffsets << <blocks, threads >> > (n, d_worker, d_blockOffsets);
            cudaDeviceSynchronize();
            checkCUDAError("kernAddBlockOffsets");

            // Phase 4: convert inclusive to exclusive 
            kernInclusiveToExclusive << <blocks, threads >> > (n, d_worker, d_output);
            cudaDeviceSynchronize();
            checkCUDAError("kernInclusiveToExclusive");
            
            timer().endGpuTimer();

            
            // copy result back
            cudaMemcpy(odata, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy device->host (naive block-shared scan)");

            // cleanup
            cudaFree(d_worker);
            cudaFree(d_output);
            cudaFree(d_blockOffsets);
            cudaFree(d_blockSums);

        }
    }
}