extern "C" __global__ void saxpy0(float* Z, float A, float* X) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  Z[id] = A * X[id];
}

extern "C" __global__ void saxpy1(float* Z, float* Y) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  Z[id] += Y[id];
}

