#include <brisbane/brisbane_openmp.h>

static void saxpy0(float* Z, float A, float* X, BRISBANE_OPENMP_KERNEL_ARGS) {
  size_t _id;
#pragma omp parallel for shared(Z, A, X) private(_id)
  BRISBANE_OPENMP_KERNEL_BEGIN
  Z[_id] = A * X[_id];
  BRISBANE_OPENMP_KERNEL_END
}

static void saxpy1(float* Z, float* Y, BRISBANE_OPENMP_KERNEL_ARGS) {
  size_t _id;
#pragma omp parallel for shared(Z, Y) private(_id)
  BRISBANE_OPENMP_KERNEL_BEGIN
  Z[_id] += Y[_id];
  BRISBANE_OPENMP_KERNEL_END
}

