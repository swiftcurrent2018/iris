#include "brisbane_poly_kernel.h"

__kernel void vecadd(__global int* restrict C, __global int* restrict A, __global int* restrict B, BRISBANE_KERNEL_ARGS) {
  BRISBANE_POLY_KERNEL_BEGIN;
  size_t i = get_global_id(0);
  A[i] = i;
  C[i] = A[i] + B[i];
  BRISBANE_POLY_KERNEL_END;
}

__kernel void saxpy(__global float* restrict Z, float A, __global float* restrict X, __global float* restrict Y, BRISBANE_KERNEL_ARGS) {
  BRISBANE_POLY_KERNEL_BEGIN;
  size_t id = get_global_id(0);
  Z[id] = A * X[id] + Y[id];
  BRISBANE_POLY_KERNEL_END;
}

