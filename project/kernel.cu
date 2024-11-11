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

__device__ int warp_sum(int val) {
  for (int i = warpSize / 2; i > 0; i = i / 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, i);
  return val;
}

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

    float tmp = account[(j - 1) * clients + i] + changes[j * clients + i];

    account[j * clients + i] = tmp;
    // Compute warp-based summation for this period
    int period_result = warp_sum(tmp);

    // Only the first thread in each warp adds the result to sum[j]
    if (warp_lane == 0) {
      atomicAdd(&sum[j], period_result);
    }
  }
}

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

void solveGPU(int *changes, int *account, int *sum, int clients, int periods) {
  // TODO put kernel call here
  //   dim3 blocks = dim3(clients / 32, periods / 32, 1);
  //   dim3 threads = dim3(32, 32, 1);
  // test2<<<blocks, threads>>>(changes, account, sum, clients, periods);
  // test1<<<clients / 32, 32>>>(changes, account, sum, clients, periods);

  test3<<<clients / 32, 32>>>(changes, account, sum, clients, periods);
}
