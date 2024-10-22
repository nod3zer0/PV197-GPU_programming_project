// TODO put kernel implementation here

__global__ void solveGPU2(int *changes, int *account, int *sum, int clients,
                          int periods) {
  for (int i = 0; i < clients; i++)
    account[i] = changes[i]; // the first change is copied
  for (int j = 1; j < periods; j++) {
    for (int i = 0; i < clients; i++) {
      account[j * clients + i] =
          account[(j - 1) * clients + i] + changes[j * clients + i];
    }
  }

  for (int j = 0; j < periods; j++) {
    int s = 0;
    for (int i = 0; i < clients; i++) {
      s += account[j * clients + i];
    }
    sum[j] = s;
  }
}

__global__ void test1(int *changes, int *account, int *sum, int clients,
                      int periods) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  account[i] = changes[i];
  atomicAdd(&sum[0], account[i]);

  for (int j = 1; j < periods; j++) {
    account[j * clients + i] =
        account[(j - 1) * clients + i] + changes[j * clients + i];

    //  sum[j] += account[j * clients + i];
    atomicAdd(&sum[j], account[j * clients + i]);
    // if (i == 0 && (j == 1 || j == 2 || j == 3 || j == 4)) {
    //   printf("account[%d] = %d\n", j * clients + i, account[j * clients +
    //   i]); printf("changes[%d] = %d\n", j * clients + i, changes[j * clients
    //   + i]);
    // }

    // printf("account[%d] = %d\n", j * clients + i, account[j * clients + i]);
  }

  __syncthreads();

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
  test1<<<clients / 32, 32>>>(changes, account, sum, clients, periods);
  //    solveGPU2<<<1, 1>>>(changes, account, sum, clients, periods);
}
