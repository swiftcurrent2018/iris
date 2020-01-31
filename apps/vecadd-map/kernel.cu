extern "C" __global__ void loop0(int* C, int* A, int* B) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  C[id] = A[id] + B[id];
}

extern "C" __global__ void loop1(int* D, int* C) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  D[id] = C[id] * 10;
}

extern "C" __global__ void loop2(int* E, int* D) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  E[id] = D[id] * 2;
}

