#include "brisbane_poly_kernel.h"

__kernel void vecadd(__global int* restrict C, __global int* restrict A, __global int* restrict B, BRISBANE_KERNEL_TAIL) {
  BRISBANE_POLY_KERNEL_BEGIN;
  size_t i = get_global_id(0);
  C[i] = A[i] + B[i];
  BRISBANE_POLY_KERNEL_END;
}

