__kernel void vecadd(__global int* restrict C, __global int* restrict A, __global int* restrict B) {
  size_t i = get_global_id(0);
  C[i] = A[i] + B[i];
}

