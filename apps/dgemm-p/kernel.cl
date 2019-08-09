#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void ijk(__global double* restrict C, __global double* restrict A, __global double* restrict B, int SIZE) {
  int i = get_global_id(1);
  int j = get_global_id(0);

  double sum = 0.0;
  for (int k = 0; k < SIZE; k++) {
    sum += A[i * SIZE + k] * B[k * SIZE + j];
  }
  C[i * SIZE + j] = sum;
}
