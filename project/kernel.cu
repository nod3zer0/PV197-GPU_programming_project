

// using shfl down to sum values in paralel by threads that are in the same warp
// (warp sum algorithm)
__device__ float warp_sum(float val) {
  val += __shfl_down_sync(0xFFFFFFFF, val, 16);
  val += __shfl_down_sync(0xFFFFFFFF, val, 8);
  val += __shfl_down_sync(0xFFFFFFFF, val, 4);
  val += __shfl_down_sync(0xFFFFFFFF, val, 2);
  val += __shfl_down_sync(0xFFFFFFFF, val, 1);
  return val;
}

// size of registers used for preloading values
#define REG_SIZE 16

// compute values
__global__ void compute(int *changes, int *account, int *sum, int clients,
                        int periods) {
  // thread index
  register int i = blockIdx.x * blockDim.x + threadIdx.x;

  // 0 account
  register int a = 0;

  // forced compiler to use registers (arrays did not work sometimes)

  // account
  register int a1, a2, a3, a4, a5, a6, a7, a8;
  // changes
  register int c1, c2, c3, c4, c5, c6, c7, c8;
  // period results (summation)
  register int period_result1, period_result2, period_result3, period_result4,
      period_result5, period_result6, period_result7, period_result8;
  // sum indexes
  register int sum_idx1, sum_idx2, sum_idx3, sum_idx4, sum_idx5, sum_idx6,
      sum_idx7, sum_idx8;

  // index of the first element for this thread
  register int idx = i;

  // temporary array for summing values
  __shared__ int period_tmp[REG_SIZE];

  // tmp arrays for preloading values
  int changes_reg[REG_SIZE];
  int account_reg[REG_SIZE];

  // main loop
  for (int j = 0; j < periods; j += REG_SIZE) {

    // preloading values into registers
    int idx_bac1 = idx;
#pragma unroll 16
    for (int k = 0; k < REG_SIZE; k++) {
      changes_reg[k] = changes[idx];
      idx += clients;
    }
    idx = idx_bac1;

    // rolled out agorihtm for computing assignment (pragma did not work)
    for (int k = 0; k < REG_SIZE; k += 8) {

      // load values into registers to help compiler optimize instructions
      c1 = changes_reg[k];
      c2 = changes_reg[k + 1];
      c3 = changes_reg[k + 2];
      c4 = changes_reg[k + 3];
      c5 = changes_reg[k + 4];
      c6 = changes_reg[k + 5];
      c7 = changes_reg[k + 6];
      c8 = changes_reg[k + 7];

      // load values into registers to help compiler optimize instructions
      a1 = a + c1;
      a2 = a1 + c2;
      a3 = a2 + c3;
      a4 = a3 + c4;
      a5 = a4 + c5;
      a6 = a5 + c6;
      a7 = a6 + c7;
      a8 = a7 + c8;

      // load values from local registers into other registers used for
      // preloading
      account_reg[k] = a1;
      account_reg[k + 1] = a2;
      account_reg[k + 2] = a3;
      account_reg[k + 3] = a4;
      account_reg[k + 4] = a5;
      account_reg[k + 5] = a6;
      account_reg[k + 6] = a7;
      account_reg[k + 7] = a8;
      // sum values in parallel by using warp_sum algorithm
      period_result1 = warp_sum(a1);
      period_result2 = warp_sum(a2);
      period_result3 = warp_sum(a3);
      period_result4 = warp_sum(a4);
      period_result5 = warp_sum(a5);
      period_result6 = warp_sum(a6);
      period_result7 = warp_sum(a7);
      period_result8 = warp_sum(a8);

      // first thread in warp will sum values into period_tmp array
      // this is done because warp is usually 32 threads so in a block there
      // are 4 warps so we can reduce number of conflicts whe doing atomicAdd
      if (threadIdx.x % warpSize == 0) {
        atomicAdd(&period_tmp[k], period_result1);
        atomicAdd(&period_tmp[k + 1], period_result2);
        atomicAdd(&period_tmp[k + 2], period_result3);
        atomicAdd(&period_tmp[k + 3], period_result4);
        atomicAdd(&period_tmp[k + 4], period_result5);
        atomicAdd(&period_tmp[k + 5], period_result6);
        atomicAdd(&period_tmp[k + 6], period_result7);
        atomicAdd(&period_tmp[k + 7], period_result8);
      }
      // update index
      a = a8;
    }

    // loading values back into global memory from registers
#pragma unroll 16
    for (int k = 0; k < REG_SIZE; k++) {
      account[idx] = account_reg[k];
      idx += clients;
    }

    // synchronize threads to make sure threads are done with their work before final summation into global memory
    __syncthreads();
    // sum values into global memory
    for (int k = 0; k < REG_SIZE; k += 8) {

      if (threadIdx.x == 0) {

        // precompute indexes
        sum_idx1 = k + j;
        sum_idx2 = k + j + 1;
        sum_idx3 = k + j + 2;
        sum_idx4 = k + j + 3;
        sum_idx5 = k + j + 4;
        sum_idx6 = k + j + 5;
        sum_idx7 = k + j + 6;
        sum_idx8 = k + j + 7;

        // sum values into global memory
        atomicAdd(&sum[sum_idx1], period_tmp[k]);
        atomicAdd(&sum[sum_idx2], period_tmp[k + 1]);
        atomicAdd(&sum[sum_idx3], period_tmp[k + 2]);
        atomicAdd(&sum[sum_idx4], period_tmp[k + 3]);
        atomicAdd(&sum[sum_idx5], period_tmp[k + 4]);
        atomicAdd(&sum[sum_idx6], period_tmp[k + 5]);
        atomicAdd(&sum[sum_idx7], period_tmp[k + 6]);
        atomicAdd(&sum[sum_idx8], period_tmp[k + 7]);

        // reset temporary array
        period_tmp[k] = 0;
        period_tmp[k + 1] = 0;
        period_tmp[k + 2] = 0;
        period_tmp[k + 3] = 0;
        period_tmp[k + 4] = 0;
        period_tmp[k + 5] = 0;
        period_tmp[k + 6] = 0;
        period_tmp[k + 7] = 0;
      }
    }
  }
}

void solveGPU(int *changes, int *account, int *sum, int clients, int periods) {

  compute<<<clients / 128, 128>>>(changes, account, sum, clients, periods);
}
