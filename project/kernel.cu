

__device__ float warp_sum(float val) {
  val += __shfl_down_sync(0xFFFFFFFF, val, 16);
  val += __shfl_down_sync(0xFFFFFFFF, val, 8);
  val += __shfl_down_sync(0xFFFFFFFF, val, 4);
  val += __shfl_down_sync(0xFFFFFFFF, val, 2);
  val += __shfl_down_sync(0xFFFFFFFF, val, 1);
  return val;
}


__global__ void final2(int *changes, int *account, int *sum, int clients,
                       int periods) {
  // thread index
 register int i = blockIdx.x * blockDim.x + threadIdx.x;

 register int a = 0;

  register int a1, a2, a3, a4, a5, a6, a7, a8;
  register int c1, c2, c3, c4, c5, c6, c7, c8;
  register int idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8;
  register int period_result1, period_result2, period_result3, period_result4,
      period_result5, period_result6, period_result7, period_result8;
  register int sum_idx1, sum_idx2, sum_idx3, sum_idx4, sum_idx5, sum_idx6,
      sum_idx7, sum_idx8;

  register int idx = i;
  register int idx_bac = 0;

  for (int k = 0; k < periods; k += 8) {



    idx1 = idx;
    idx2 = idx1 + clients;
    idx3 = idx2 + clients;
    idx4 = idx3 + clients;
    idx5 = idx4 + clients;
    idx6 = idx5 + clients;
    idx7 = idx6 + clients;
    idx8 = idx7 + clients;

    c1 = changes[idx1];
    c2 = changes[idx2];
    c3 = changes[idx3];
    c4 = changes[idx4];
    c5 = changes[idx5];
    c6 = changes[idx6];
    c7 = changes[idx7];
    c8 = changes[idx8];

    a1 = a + c1;
    a2 = a1 + c2;
    a3 = a2 + c3;
    a4 = a3 + c4;
    a5 = a4 + c5;
    a6 = a5 + c6;
    a7 = a6 + c7;
    a8 = a7 + c8;

    account[idx1] = a1;
    account[idx2] = a2;
    account[idx3] = a3;
    account[idx4] = a4;
    account[idx5] = a5;
    account[idx6] = a6;
    account[idx7] = a7;
    account[idx8] = a8;


    period_result1 = warp_sum(a1);
    period_result2 = warp_sum(a2);
    period_result3 = warp_sum(a3);
    period_result4 = warp_sum(a4);
    period_result5 = warp_sum(a5);
    period_result6 = warp_sum(a6);
    period_result7 = warp_sum(a7);
    period_result8 = warp_sum(a8);

    sum_idx1 = k;
    sum_idx2 = k + 1;
    sum_idx3 = k + 2;
    sum_idx4 = k + 3;
    sum_idx5 = k + 4;
    sum_idx6 = k + 5;
    sum_idx7 = k + 6;
    sum_idx8 = k + 7;

    if (threadIdx.x % warpSize == 0) {
      atomicAdd(&sum[sum_idx8], period_result8);
      atomicAdd(&sum[sum_idx7], period_result7);
      atomicAdd(&sum[sum_idx6], period_result6);
      atomicAdd(&sum[sum_idx5], period_result5);
      atomicAdd(&sum[sum_idx4], period_result4);
      atomicAdd(&sum[sum_idx3], period_result3);
      atomicAdd(&sum[sum_idx2], period_result2);
      atomicAdd(&sum[sum_idx1], period_result1);
    }

    a = a8;
    idx += 8 * clients;
  }
}



void solveGPU(int *changes, int *account, int *sum, int clients, int periods) {
  // TODO put kernel call here
  //   dim3 blocks = dim3(clients / 32, periods / 32, 1);
  //   dim3 threads = dim3(32, 32, 1);
  // test2<<<blocks, threads>>>(changes, account, sum, clients, periods);
  // test1<<<clients / 32, 32>>>(changes, account, sum, clients, periods);

  // test0<<<clients / 128, 128>>>(changes, account, sum, clients, periods);
  final2<<<clients / 128, 128>>>(changes, account, sum, clients, periods);
}
