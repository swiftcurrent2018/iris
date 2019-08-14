#include <brisbane/brisbane.h>
#include <brisbane/brisbane_poly_types.h>
#include <brisbane/brisbane_poly.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
/* __kernel void ijk(__global double* restrict C, __global double* restrict A, __global double* restrict B, int SIZE) {
  int i = get_global_id(1);
  int j = get_global_id(0);

  double sum = 0.0;
  for (int k = 0; k < SIZE; k++) {
    sum += A[i * SIZE + k] * B[k * SIZE + j];
  }
  C[i * SIZE + j] = sum;
} */

typedef struct {
  brisbane_poly_mem C;
  brisbane_poly_mem A;
  brisbane_poly_mem B;
  int SIZE;
} brisbane_poly_ijk_args;
brisbane_poly_ijk_args ijk_args;

int brisbane_poly_ijk_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    case 3: memcpy(&ijk_args.SIZE, value, size); break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

int brisbane_poly_ijk_getmem(int idx, brisbane_poly_mem* mem) {
  switch (idx) {
    case 0: memcpy(mem, &ijk_args.C, sizeof(brisbane_poly_mem)); break;
    case 1: memcpy(mem, &ijk_args.A, sizeof(brisbane_poly_mem)); break;
    case 2: memcpy(mem, &ijk_args.B, sizeof(brisbane_poly_mem)); break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}
#include "kernel-poly.h"

int brisbane_poly_kernel(const char* name) {
  brisbane_poly_lock();
  if (strcmp(name, "ijk") == 0) {
    brisbane_poly_kernel_idx = 0;
    return BRISBANE_OK;
  }
  return BRISBANE_ERR;
}

int brisbane_poly_setarg(int idx, size_t size, void* value) {
  switch (brisbane_poly_kernel_idx) {
    case 0: return brisbane_poly_ijk_setarg(idx, size, value);
  }
  return BRISBANE_ERR;
}

int brisbane_poly_launch(int dim, size_t* wgo, size_t* wgs, size_t* gws, size_t* lws) {
  int ret = BRISBANE_OK;
  switch (brisbane_poly_kernel_idx) {
    case 0: ret = ijk(wgo[0], wgo[1], wgo[2], wgs[0], wgs[1], wgs[2], gws[0], gws[1], gws[2], lws[0], lws[1], lws[2]); break;
  }
  brisbane_poly_unlock();
  return ret;
}

int brisbane_poly_getmem(int idx, brisbane_poly_mem* mem) {
  switch (brisbane_poly_kernel_idx) {
    case 0: return brisbane_poly_ijk_getmem(idx, mem);
  }
  return BRISBANE_ERR;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif

