#include <brisbane/brisbane_omp.h>

#ifdef __cplusplus
extern "C" {
#endif

static int brisbane_omp_kernel_idx;

typedef struct {
  float* Z;
  float A;
  float* X;
  float* Y;
} brisbane_omp_saxpy_args;
brisbane_omp_saxpy_args saxpy_args;

static int brisbane_omp_saxpy_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    case 1: memcpy(&saxpy_args.A, value, size); break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

static int brisbane_omp_saxpy_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: saxpy_args.Z = (float*) mem; break;
    case 2: saxpy_args.X = (float*) mem; break;
    case 3: saxpy_args.Y = (float*) mem; break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

static void saxpy(float* Z, float A, float* X, float* Y, BRISBANE_OMP_KERNEL_ARGS) {
  size_t _id;
#pragma omp parallel for shared(Z, A, X, Y) private(_id)
  BRISBANE_OMP_KERNEL_BEGIN
  Z[_id] = A * X[_id] + Y[_id];
  BRISBANE_OMP_KERNEL_END
}

int brisbane_omp_kernel(const char* name) {
  if (strcmp(name, "saxpy") == 0) {
    brisbane_omp_kernel_idx = 0;
    return BRISBANE_OK;
  }
  return BRISBANE_ERR;
}

int brisbane_omp_setarg(int idx, size_t size, void* value) {
  switch (brisbane_omp_kernel_idx) {
    case 0: return brisbane_omp_saxpy_setarg(idx, size, value);
  }
  return BRISBANE_ERR;
}

int brisbane_omp_setmem(int idx, void* mem) {
  switch (brisbane_omp_kernel_idx) {
    case 0: return brisbane_omp_saxpy_setmem(idx, mem);
  }
  return BRISBANE_ERR;
}

int brisbane_omp_launch(int dim, size_t off, size_t ndr) {
  int ret = BRISBANE_OK;
  switch (brisbane_omp_kernel_idx) {
    case 0: saxpy(saxpy_args.Z, saxpy_args.A, saxpy_args.X, saxpy_args.Y, off, ndr); break;
  }
  return ret;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif

