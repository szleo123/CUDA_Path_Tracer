#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/detail/raw_pointer_cast.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            if (n <= 0) {
                return;
            }


            // copy input to device
            thrust::device_vector<int> d_in(idata, idata + n);

            // output device vector
            thrust::device_vector<int> d_out(n);

            // exclusive scan on device
            timer().startGpuTimer();
            thrust::exclusive_scan(d_in.begin(), d_in.end(), d_out.begin());
            timer().endGpuTimer();

            // copy result back to host
            cudaMemcpy(odata, thrust::raw_pointer_cast(d_out.data()), n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("thrust::scan cudaMemcpy");

        }
    }
}