#include <hip/hip_runtime.h>

extern "C" __global__ void saxpy0(float* Z, float A, float* X, size_t blockOff_x) {
  size_t id = (blockOff_x + blockIdx.x) * blockDim.x + threadIdx.x;
  Z[id] = A * X[id];
}

extern "C" __global__ void saxpy1(float* Z, float* Y, size_t blockOff_x) {
  size_t id = (blockOff_x + blockIdx.x) * blockDim.x + threadIdx.x;
  Z[id] += Y[id];
}

