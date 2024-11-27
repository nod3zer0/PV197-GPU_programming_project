

// using shfl down to sum values in paralel by threads that are in the same warp
__device__ float warp_sum(float val) {
  val += __shfl_down_sync(0xFFFFFFFF, val, 16);
  val += __shfl_down_sync(0xFFFFFFFF, val, 8);
  val += __shfl_down_sync(0xFFFFFFFF, val, 4);
  val += __shfl_down_sync(0xFFFFFFFF, val, 2);
  val += __shfl_down_sync(0xFFFFFFFF, val, 1);
  return val;
}

// compute values
__global__ void compute(int *changes, int *account, int *sum, int clients,
                        int periods) {
  // thread index
  register int i = blockIdx.x * blockDim.x + threadIdx.x;

  register int a = 0;

  // forced compiler to use registers (arrays did not work sometimes)

  // account
  register int a1, a2, a3, a4, a5, a6, a7, a8;
  // changes
  register int c1, c2, c3, c4, c5, c6, c7, c8;
  // account indexes
  register int idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8;
  // period results (summation)
  register int period_result1, period_result2, period_result3, period_result4,
      period_result5, period_result6, period_result7, period_result8;
  // sum indexes
  register int sum_idx1, sum_idx2, sum_idx3, sum_idx4, sum_idx5, sum_idx6,
      sum_idx7, sum_idx8;

  // index of the first element for this thread
  register int idx = i;
  register int idx_bac = 0;

  // temporary array for summing values
  __shared__ int period_tmp[8];



  // rolled out agorihtm for computing assignment (pragma did not work)
  for (int k = 0; k < periods; k += 8) {

    // precompute indexes
    idx1 = idx;
    idx2 = idx1 + clients;
    idx3 = idx2 + clients;
    idx4 = idx3 + clients;
    idx5 = idx4 + clients;
    idx6 = idx5 + clients;
    idx7 = idx6 + clients;
    idx8 = idx7 + clients;

    // load values into registers to help compiler optimize instructions
    c1 = changes[idx1];
    c2 = changes[idx2];
    c3 = changes[idx3];
    c4 = changes[idx4];
    c5 = changes[idx5];
    c6 = changes[idx6];
    c7 = changes[idx7];
    c8 = changes[idx8];

    // load values into registers to help compiler optimize instructions
    a1 = a + c1;
    a2 = a1 + c2;
    a3 = a2 + c3;
    a4 = a3 + c4;
    a5 = a4 + c5;
    a6 = a5 + c6;
    a7 = a6 + c7;
    a8 = a7 + c8;

    // load values back from registers to global memory
    account[idx1] = a1;
    account[idx2] = a2;
    account[idx3] = a3;
    account[idx4] = a4;
    account[idx5] = a5;
    account[idx6] = a6;
    account[idx7] = a7;
    account[idx8] = a8;


    // sum values in parallel by using warp_sum algorithm
    period_result1 = warp_sum(a1);
    period_result2 = warp_sum(a2);
    period_result3 = warp_sum(a3);
    period_result4 = warp_sum(a4);
    period_result5 = warp_sum(a5);
    period_result6 = warp_sum(a6);
    period_result7 = warp_sum(a7);
    period_result8 = warp_sum(a8);

    // precompute indexes
    sum_idx1 = k;
    sum_idx2 = k + 1;
    sum_idx3 = k + 2;
    sum_idx4 = k + 3;
    sum_idx5 = k + 4;
    sum_idx6 = k + 5;
    sum_idx7 = k + 6;
    sum_idx8 = k + 7;

    // first thread in warp will sum values into period_tmp array
    // this is done becouse warp is ususally 32 threads so in a block there are 4 warps so we can reduce number of conflicts whe doing atomicAdd
    if (threadIdx.x % warpSize == 0) {
      atomicAdd(&period_tmp[0], period_result1);
      atomicAdd(&period_tmp[1], period_result2);
      atomicAdd(&period_tmp[2], period_result3);
      atomicAdd(&period_tmp[3], period_result4);
      atomicAdd(&period_tmp[4], period_result5);
      atomicAdd(&period_tmp[5], period_result6);
      atomicAdd(&period_tmp[6], period_result7);
      atomicAdd(&period_tmp[7], period_result8);
    }
    // sync threads before final atomicAdd into global memory
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(&sum[sum_idx1], period_tmp[0]);
      atomicAdd(&sum[sum_idx2], period_tmp[1]);
      atomicAdd(&sum[sum_idx3], period_tmp[2]);
      atomicAdd(&sum[sum_idx4], period_tmp[3]);
      atomicAdd(&sum[sum_idx5], period_tmp[4]);
      atomicAdd(&sum[sum_idx6], period_tmp[5]);
      atomicAdd(&sum[sum_idx7], period_tmp[6]);
      atomicAdd(&sum[sum_idx8], period_tmp[7]);

        // reset temporary array
      period_tmp[0] = 0;
      period_tmp[1] = 0;
      period_tmp[2] = 0;
      period_tmp[3] = 0;
      period_tmp[4] = 0;
      period_tmp[5] = 0;
      period_tmp[6] = 0;
      period_tmp[7] = 0;
    }

    // update index
    a = a8;
    idx += 8 * clients;
  }
}

void solveGPU(int *changes, int *account, int *sum, int clients, int periods) {

  compute<<<clients / 128, 128>>>(changes, account, sum, clients, periods);
}
