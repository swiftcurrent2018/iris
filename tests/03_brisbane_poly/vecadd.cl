__kernel void vecadd(__global int* restrict C, __global int* restrict A, __global int* restrict B) {
  size_t i = get_global_id(0);
  C[i] = A[i] + B[i];
}

__kernel void saxpy(__global float* restrict Z, float A, __global float* restrict X, __global float* restrict Y) {
  size_t i = get_global_id(0);
  Z[i] = A * X[i] + Y[i];
}

