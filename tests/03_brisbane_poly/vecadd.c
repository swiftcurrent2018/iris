#include "brisbane_poly_kernel.h"

__kernel void vecadd(__global int* restrict C, __global int* restrict A, __global int* restrict B, int AI, BRISBANE_POLY_KERNEL_ARGS) {
  BRISBANE_POLY_KERNEL_BEGIN;
  size_t i = get_global_id(0);
  C[i] = A[i] + B[i];
  BRISBANE_POLY_KERNEL_END;
}

__kernel void saxpy(__global float* restrict Z, float A, __global float* restrict X, __global float* restrict Y, BRISBANE_POLY_KERNEL_ARGS) {
  BRISBANE_POLY_KERNEL_BEGIN;
  size_t i = get_global_id(0);
  Z[i] = A * X[i] + Y[i + (int) X[i]];
  BRISBANE_POLY_KERNEL_END;
}
