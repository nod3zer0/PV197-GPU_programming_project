// TODO put kernel implementation here

// first deadline submission
// each user is handled by a single thread
__global__ void test1(int *changes, int *account, int *sum, int clients,
                      int periods) {
  // thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // copy the first change
  account[i] = changes[i];
  // atomically add from each user to the sum (not ideal, will be fixed by next
  // deadline)
  atomicAdd(&sum[0], account[i]);

  // main loop
  for (int j = 1; j < periods; j++) {

    account[j * clients + i] =
        account[(j - 1) * clients + i] + changes[j * clients + i];
    atomicAdd(&sum[j], account[j * clients + i]);
  }
}

__device__ float warp_sum(float val) {
  for (int i = warpSize / 2; i > 0; i = i / 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, i);
  return val;
}

__global__ void test0(int *changes, int *account, int *sum, int clients,
                      int periods) {
  // thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // copy the first change
  account[i] = changes[i];
  int a = account[i];
  // atomically add from each user to the sum (not ideal, will be fixed by next
  // deadline)
  // atomicAdd(&sum[0], account[i]);

  int result = warp_sum(account[i]);

  // Perform warp-level sum and only the first thread in each warp
  // will perform an atomicAdd to sum[0].
  int warp_lane = threadIdx.x % warpSize;

  if (warp_lane == 0) {
    atomicAdd(&sum[0], result);
  }

  // main loop
  int idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8;
  int b1, b2, b3, b4, b5, b6, b7, b8;
  int sum_idx1, sum_idx2, sum_idx3, sum_idx4, sum_idx5, sum_idx6, sum_idx7,
      sum_idx8;
  int period_result1, period_result2, period_result3, period_result4,
      period_result5, period_result6, period_result7, period_result8;

  int idx = i;
  int j = 1;
  for (; j < periods - 8; j += 8) { // Unroll by a factor of 8
    // Compute indices for the unrolled iterations
    idx1 = idx + clients;
    idx2 = idx1 + clients;
    idx3 = idx2 + clients;
    idx4 = idx3 + clients;
    idx5 = idx4 + clients;
    idx6 = idx5 + clients;
    idx7 = idx6 + clients;
    idx8 = idx7 + clients;

    // Load changes for unrolled iterations
    b1 = changes[idx1];
    b2 = changes[idx2];
    b3 = changes[idx3];
    b4 = changes[idx4];
    b5 = changes[idx5];
    b6 = changes[idx6];
    b7 = changes[idx7];
    b8 = changes[idx8];

    // Compute and store results

    a += b1;
    account[idx1] = a;
    period_result1 = warp_sum(a);

    a += b2;
    account[idx2] = a;
    period_result2 = warp_sum(a);

    a += b3;
    account[idx3] = a;
    period_result3 = warp_sum(a);

    a += b4;
    account[idx4] = a;
    period_result4 = warp_sum(a);

    a += b5;
    account[idx5] = a;
    period_result5 = warp_sum(a);

    a += b6;
    account[idx6] = a;
    period_result6 = warp_sum(a);

    a += b7;
    account[idx7] = a;
    period_result7 = warp_sum(a);

    a += b8;
    account[idx8] = a;
    period_result8 = warp_sum(a);

    sum_idx1 = j;
    sum_idx2 = j + 1;
    sum_idx3 = j + 2;
    sum_idx4 = j + 3;
    sum_idx5 = j + 4;
    sum_idx6 = j + 5;
    sum_idx7 = j + 6;
    sum_idx8 = j + 7;

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
    // Update the index for the next unrolled iteration
    idx += 8 * clients;
  }

  //  idx -= 8 * clients;
  // Compute indices for the unrolled iterations
  idx1 = idx + clients;
  idx2 = idx1 + clients;
  idx3 = idx2 + clients;
  idx4 = idx3 + clients;
  idx5 = idx4 + clients;
  idx6 = idx5 + clients;
  idx7 = idx6 + clients;

  // Load changes for unrolled iterations
  b1 = changes[idx1];
  b2 = changes[idx2];
  b3 = changes[idx3];
  b4 = changes[idx4];
  b5 = changes[idx5];
  b6 = changes[idx6];
  b7 = changes[idx7];

  // Compute and store results

  a += b1;
  account[idx1] = a;
  period_result1 = warp_sum(a);

  a += b2;
  account[idx2] = a;
  period_result2 = warp_sum(a);

  a += b3;
  account[idx3] = a;
  period_result3 = warp_sum(a);

  a += b4;
  account[idx4] = a;
  period_result4 = warp_sum(a);

  a += b5;
  account[idx5] = a;
  period_result5 = warp_sum(a);

  a += b6;
  account[idx6] = a;
  period_result6 = warp_sum(a);

  a += b7;
  account[idx7] = a;
  period_result7 = warp_sum(a);

  sum_idx1 = j;
  sum_idx2 = j + 1;
  sum_idx3 = j + 2;
  sum_idx4 = j + 3;
  sum_idx5 = j + 4;
  sum_idx6 = j + 5;
  sum_idx7 = j + 6;

  if (threadIdx.x % warpSize == 0) {

    atomicAdd(&sum[sum_idx7], period_result7);
    atomicAdd(&sum[sum_idx6], period_result6);
    atomicAdd(&sum[sum_idx5], period_result5);
    atomicAdd(&sum[sum_idx4], period_result4);
    atomicAdd(&sum[sum_idx3], period_result3);
    atomicAdd(&sum[sum_idx2], period_result2);
    atomicAdd(&sum[sum_idx1], period_result1);
  }
}

//---------------experimental code-------------------

// __device__ int warp_sum(int val) {
//   for (int offset = 1; offset < warpSize; offset *= 2) {
//     int val += __shfl_up_sync(0xFFFFFFFF, val, offset);
//     if (threadIdx.x % warpSize >= offset) {
//       val += temp;
//     }
//   }
//   return val;
// }

// __device__ int warp_sum(int val) {
//   for (int i = 1; i < warpSize; i *= 2)
//     val += __shfl_xor_sync(-1, val, i);
//   return val;
// }

__global__ void test3(int *changes, int *account, int *sum, int clients,
                      int periods) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Initialize account with initial changes
  account[i] = changes[i];
  int result = warp_sum(account[i]);

  // Perform warp-level sum and only the first thread in each warp
  // will perform an atomicAdd to sum[0].
  int warp_lane = threadIdx.x % warpSize;

  if (warp_lane == 0) {
    atomicAdd(&sum[0], result);
  }

  // Compute the running account balance for each period
  for (int j = 1; j < periods; j++) {
    // Update account with cumulative sum for this period

    int tmp = account[(j - 1) * clients + i] + changes[j * clients + i];

    account[j * clients + i] = tmp;
    // Compute warp-based summation for this period
    int period_result = warp_sum(tmp);

    // Only the first thread in each warp adds the result to sum[j]
    if (warp_lane == 0) {
      atomicAdd(&sum[j], period_result);
    }
  }
}

// https://users.umiacs.umd.edu/~ramani/cmsc828e_gpusci/ScanTalk.pdf

// __global__ void prescan(float *g_odata, float *g_idata, int n) {
//   extern __shared__ float temp[];
//   int thid = threadIdx.x;
//   int offset = 1;
//   temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory
//   temp[2 * thid + 1] = g_idata[2 * thid + 1];
//   for (int d = n >> 1; d > 0; d >> = 1)
//   // build sum in place up the tree
//   {
//     __syncthreads();
//     if (thid < d) {

//       int ai = offset * (2 * thid + 1) - 1;
//       int bi = offset * (2 * thid + 2) - 1;

//       temp[bi] += temp[ai];
//     }
//     offset *= 2;
//   }
//   if (thid == 0) {
//     temp[n - 1] = 0;
//   } // clear the last element
//   for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
//   {
//     offset >> = 1;
//     __syncthreads();
//     if (thid < d) {

//       int ai = offset * (2 * thid + 1) - 1;
//       int bi = offset * (2 * thid + 2) - 1;
//       float t = temp[ai];
//       temp[ai] = temp[bi];
//       temp[bi] += t;
//     }
//   }
//   __syncthreads();

//   g_odata[2 * thid] = temp[2 * thid]; // write results to device memory
//   g_odata[2 * thid + 1] = temp[2 * thid + 1];
// }

//-----------------------------------------

// second prototype currently not working
__global__ void test2(int *changes, int *account, int *sum, int clients,
                      int periods) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  //   if (y == 0) {
  //     atomicAdd(&account[i], changes[i]);
  //     return;
  //   }

  for (int j = y; j < (blockIdx.y + 1) * blockDim.y; j++) {

    // if (j == 1 && i == 0) {
    //   printf("changes[%d] = %d\n", j * clients + i, changes[j * clients +
    //   i]); printf("account[%d] = %d\n",
    //          account[(j - 1) * clients + i] + changes[j * clients + i]);
    // }

    atomicAdd(&account[j * clients + i], changes[y * clients + i]);
  }

  //   for (int j = 1; j < periods; j++) {
  //     account[j * clients + i] =
  //         account[(j - 1) * clients + i] + changes[j * clients + i];

  //     //  sum[j] += account[j * clients + i];
  //     if (y == 0) {
  //       atomicAdd(&sum[j], account[j * clients + i]);
  //     }

  //     // if (i == 0 && (j == 1 || j == 2 || j == 3 || j == 4)) {
  //     //   printf("account[%d] = %d\n", j * clients + i, account[j *
  //     clients
  //     +
  //     //   i]); printf("changes[%d] = %d\n", j * clients + i, changes[j *
  //     //   clients
  //     //   + i]);
  //     // }

  //     // printf("account[%d] = %d\n", j * clients + i, account[j * clients
  //     +
  //     // i]);
  //   }

  //   if (i == 0) {
  //     for (int j = 0; j < periods; j++) {
  //       int s = 0;
  //       for (int k = 0; k < clients; k++) {
  //         s += account[j * clients + k];
  //       }
  //       sum[j] = s;
  //     }
  //   }
}

__global__ void prescan(int *g_odata, int *g_idata, int n) {
  extern __shared__ float temp[]; // allocated on invocation

  int thid = threadIdx.x;
  int offset = 1;

  temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory
  temp[2 * thid + 1] = g_idata[2 * thid + 1];

  for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
  {
    __syncthreads();

    if (thid < d) {
      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  if (thid == 0) {
    temp[n - 1] = 0;
  } // clear the last element

  for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
  {
    offset >>= 1;
    __syncthreads();

    if (thid < d) {
      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;

      float t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }

  __syncthreads();

  g_odata[2 * thid] = temp[2 * thid]; // write results to device memory
  g_odata[2 * thid + 1] = temp[2 * thid + 1];
}

void solveGPU(int *changes, int *account, int *sum, int clients, int periods) {
  // TODO put kernel call here
  //   dim3 blocks = dim3(clients / 32, periods / 32, 1);
  //   dim3 threads = dim3(32, 32, 1);
  // test2<<<blocks, threads>>>(changes, account, sum, clients, periods);
  // test1<<<clients / 32, 32>>>(changes, account, sum, clients, periods);

  test0<<<clients / 128, 128>>>(changes, account, sum, clients, periods);
}
