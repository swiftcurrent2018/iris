#include <brisbane/brisbane_openmp.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  __global float * Z;
  float A;
  __global float * X;
} brisbane_openmp_saxpy0_args;
brisbane_openmp_saxpy0_args saxpy0_args;

static int brisbane_openmp_saxpy0_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    case 1: memcpy(&saxpy0_args.A, value, size); break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

static int brisbane_openmp_saxpy0_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: saxpy0_args.Z = (__global float *) mem; break;
    case 2: saxpy0_args.X = (__global float *) mem; break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}
typedef struct {
  __global float * Z;
  __global float * Y;
} brisbane_openmp_saxpy1_args;
brisbane_openmp_saxpy1_args saxpy1_args;

static int brisbane_openmp_saxpy1_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

static int brisbane_openmp_saxpy1_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: saxpy1_args.Z = (__global float *) mem; break;
    case 1: saxpy1_args.Y = (__global float *) mem; break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

#include "kernel.cl.openmp.h"

int brisbane_openmp_kernel(const char* name) {
  brisbane_openmp_lock();
  if (strcmp(name, "saxpy0") == 0) {
    brisbane_openmp_kernel_idx = 0;
    return BRISBANE_OK;
  }
  if (strcmp(name, "saxpy1") == 0) {
    brisbane_openmp_kernel_idx = 1;
    return BRISBANE_OK;
  }
  return BRISBANE_ERR;
}

int brisbane_openmp_setarg(int idx, size_t size, void* value) {
  switch (brisbane_openmp_kernel_idx) {
    case 0: return brisbane_openmp_saxpy0_setarg(idx, size, value);
    case 1: return brisbane_openmp_saxpy1_setarg(idx, size, value);
  }
  return BRISBANE_ERR;
}

int brisbane_openmp_setmem(int idx, void* mem) {
  switch (brisbane_openmp_kernel_idx) {
    case 0: return brisbane_openmp_saxpy0_setmem(idx, mem);
    case 1: return brisbane_openmp_saxpy1_setmem(idx, mem);
  }
  return BRISBANE_ERR;
}

int brisbane_openmp_launch(int dim, size_t off, size_t ndr) {
  switch (brisbane_openmp_kernel_idx) {
    case 0: saxpy0(saxpy0_args.Z, saxpy0_args.A, saxpy0_args.X, off, ndr); break;
    case 1: saxpy1(saxpy1_args.Z, saxpy1_args.Y, off, ndr); break;
  }
  brisbane_openmp_unlock();
  return BRISBANE_OK;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif

