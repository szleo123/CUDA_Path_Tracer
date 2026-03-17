#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int cur_sum = 0; 
            odata[0] = cur_sum; 
            for (int i = 0; i < n - 1; i++) {
                cur_sum += idata[i]; 
                odata[i + 1] = cur_sum; 
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int cur_index = 0; 
            for (int i = 0; i < n; i++) {
                int cur_read = idata[i]; 
                if (cur_read != 0) {
                    odata[cur_index] = cur_read; 
                    cur_index++; 
                }
            }
            timer().endCpuTimer();
            return cur_index;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            if (n <= 0) {
                timer().endCpuTimer();
                return 0;
            }

            // step 1 compute temp array (mask)
            int* temp = new int[n];
            int* scan_result = new int[n];
            for (int i = 0; i < n; i++) {
                temp[i] = idata[i] != 0 ? 1 : 0;
            }

            // step 2 run exclusive scan; (scan_result is the output, temp is the input)
            int cur_sum = 0;
            scan_result[0] = cur_sum;
            for (int i = 0; i < n - 1; i++) {
                cur_sum += temp[i];
                scan_result[i + 1] = cur_sum;
            }

            // step 3 scatter
            for (int j = 0; j < n; j++) {
                if (temp[j] == 1) {
                    odata[scan_result[j]] = idata[j];
                }
            }

            // total number of 1s: last scanned value plus last mask element (for exclusive scan)
            int num = scan_result[n - 1] + temp[n - 1];

            timer().endCpuTimer();
            delete[] temp;
            delete[] scan_result;
            return num;
        }
    }
}
