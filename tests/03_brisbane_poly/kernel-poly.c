#include <brisbane/brisbane_poly_types.h>
#include <brisbane/brisbane_poly.h>

typedef struct {
  brisbane_poly_mem C;
  brisbane_poly_mem A;
  brisbane_poly_mem B;
  int AI;
  BRISBANE_POLY_KERNEL_ARGS_STRUCT;
} bp_vecadd_args;

bp_vecadd_args vecadd_args;

int vecadd(BRISBANE_POLY_KERNEL_ARGS) {
  BRISBANE_POLY_ARRAY_2D(vecadd, A, sizeof(i32), 0, _lws0);
  BRISBANE_POLY_ARRAY_2D(vecadd, B, sizeof(i32), 0, _lws0);
  BRISBANE_POLY_ARRAY_2D(vecadd, C, sizeof(i32), 0, _lws0);

  {
  BRISBANE_POLY_DOMAIN(i2, 0, -_wgo0 + _wgs0 - 1);
  BRISBANE_POLY_DOMAIN(i5, 0, _lws0 - 1);
  BRISBANE_POLY_READ(vecadd, A, _wgo0 + i2[0], i5[0]);
  BRISBANE_POLY_READ(vecadd, A, _wgo0 + i2[1], i5[1]);
  }

  {
  BRISBANE_POLY_DOMAIN(i2, 0, -_wgo0 + _wgs0 - 1);
  BRISBANE_POLY_DOMAIN(i5, 0, _lws0 - 1);
  BRISBANE_POLY_READ(vecadd, B, _wgo0 + i2[0], i5[1]);
  BRISBANE_POLY_READ(vecadd, B, _wgo0 + i2[0], i5[1]);
  }

  {
  BRISBANE_POLY_DOMAIN(i2, 0, -_wgo0 + _wgs0 - 1);
  BRISBANE_POLY_DOMAIN(i5, 0, _lws0 - 1);
  BRISBANE_POLY_MUWR(vecadd, C, _wgo0 + i2[0], i5[0]);
  BRISBANE_POLY_MUWR(vecadd, C, _wgo0 + i2[1], i5[1]);
  }

  return 0;
}

int main(int argc, char** argv) {
  size_t wgo[3] = { 0, 0, 0 };
  size_t wgs[3] = { 10, 1, 1 };
  size_t gws[3] = { 40, 1, 1 };
  size_t lws[3] = { 4, 1, 1 };

  int ret = vecadd(wgo[0], wgo[1], wgo[2], wgs[0], wgs[1], wgs[2], gws[0], gws[1], gws[2], lws[0], lws[1], lws[2]);

  printf("A TYPESZ[%lu] READ[%lu][%lu] WRITE[%lu][%lu]\n", vecadd_args.A.typesz, vecadd_args.A.r0, vecadd_args.A.r1, vecadd_args.A.w0, vecadd_args.A.w1);
  printf("B TYPESZ[%lu] READ[%lu][%lu] WRITE[%lu][%lu]\n", vecadd_args.B.typesz, vecadd_args.B.r0, vecadd_args.B.r1, vecadd_args.B.w0, vecadd_args.B.w1);
  printf("C TYPESZ[%lu] READ[%lu][%lu] WRITE[%lu][%lu]\n", vecadd_args.C.typesz, vecadd_args.C.r0, vecadd_args.C.r1, vecadd_args.C.w0, vecadd_args.C.w1);
  printf("ret[%d]\n", ret);
  return 0;
}

