#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>

typedef int32_t i32;

#define BRISBANE_POLY_KERNEL_ARGS   size_t _wgo0, size_t _wgo1, size_t _wgo2, \
                                    size_t _wgs0, size_t _wgs1, size_t _wgs2, \
                                    size_t _gws0, size_t _gws1, size_t _gws2, \
                                    size_t _lws0, size_t _lws1, size_t _lws2
#define BRISBANE_POLY_ARRAY_2D(F, M, TYPESZ, S1, S0)                          \
        F##_args.M.dim    = 2;                                                \
        F##_args.M.typesz = TYPESZ;                                           \
        F##_args.M.s1     = S1;                                               \
        F##_args.M.s0     = S0
#define BRISBANE_POLY_DOMAIN(D, I0, I1)                                       \
        size_t D[2] = { I0, I1 }       
#define BRISBANE_POLY_READ(F, M, I0, I1)                                      \
        brisbane_poly_read(&F##_args.M, I0 * F##_args.M.s0 + I1);
#define BRISBANE_POLY_MUWR(F, M, I0, I1)                                      \
        brisbane_poly_muwr(&F##_args.M, I0 * F##_args.M.s0 + I1);             

typedef struct {
  size_t typesz;
  size_t s0;
  size_t s1;
  size_t r0;
  size_t r1;
  size_t w0;
  size_t w1;
  int dim;
} brisbane_poly;

typedef struct {
  brisbane_poly C;
  brisbane_poly A;
  brisbane_poly B;
  int AI;
  size_t _wgo0; size_t _wgo1; size_t _wgo2;
  size_t _wgs0; size_t _wgs1; size_t _wgs2;
  size_t _gws0; size_t _gws1; size_t _gws2;
  size_t _lws0; size_t _lws1; size_t _lws2;
} bp_vecadd_args;

bp_vecadd_args vecadd_args;

void brisbane_poly_read(brisbane_poly* p, size_t idx) {
  if (idx < p->r0) p->r0 = idx;
  if (idx > p->r1) p->r1 = idx;
}

void brisbane_poly_muwr(brisbane_poly* p, size_t idx) {
  if (idx < p->w0) p->w0 = idx;
  if (idx > p->w1) p->w1 = idx;
}

void brisbane_poly_mawr(brisbane_poly* p, size_t idx) {
  return brisbane_poly_muwr(p, idx);
}

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

