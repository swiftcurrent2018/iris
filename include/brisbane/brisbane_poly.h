#ifndef BRISBANE_INCLUDE_BRISBANE_POLY_H
#define BRISBANE_INCLUDE_BRISBANE_POLY_H

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

typedef int32_t i32;

#define BRISBANE_POLY_KERNEL_ARGS   size_t _wgo0, size_t _wgo1, size_t _wgo2, \
                                    size_t _wgs0, size_t _wgs1, size_t _wgs2, \
                                    size_t _gws0, size_t _gws1, size_t _gws2, \
                                    size_t _lws0, size_t _lws1, size_t _lws2
#define BRISBANE_POLY_KERNEL_ARGS_STRUCT                                      \
                                    size_t _wgo0; size_t _wgo1; size_t _wgo2; \
                                    size_t _wgs0; size_t _wgs1; size_t _wgs2; \
                                    size_t _gws0; size_t _gws1; size_t _gws2; \
                                    size_t _lws0; size_t _lws1; size_t _lws2
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
#define BRISBANE_POLY_MAWR(F, M, I0, I1)                                      \
        brisbane_poly_mawr(&F##_args.M, I0 * F##_args.M.s0 + I1);             

static pthread_mutex_t brisbane_poly_mutex;
static int brisbane_poly_kernel_idx;

void brisbane_poly_init() {
  pthread_mutex_init(&brisbane_poly_mutex, NULL);
}

void brisbane_poly_finalize() {
  pthread_mutex_destroy(&brisbane_poly_mutex);
}

static void brisbane_poly_lock() {
  pthread_mutex_lock(&brisbane_poly_mutex);
}

static void brisbane_poly_unlock() {
  pthread_mutex_unlock(&brisbane_poly_mutex);
}

static void brisbane_poly_mem_init(brisbane_poly_mem* p) {
  p->r0 = SIZE_MAX;
  p->r1 = 0;
  p->w0 = SIZE_MAX;
  p->w1 = 0;
}

static void brisbane_poly_read(brisbane_poly_mem* p, size_t idx) {
  printf("[%s:%d] mem[%p] idx[%lu] r0[%lu] r1[%lu]\n", __FILE__, __LINE__, p, idx, p->r0, p->r1);
  if (idx < p->r0) p->r0 = idx;
  if (idx > p->r1) p->r1 = idx;
}

static void brisbane_poly_muwr(brisbane_poly_mem* p, size_t idx) {
  printf("[%s:%d] mem[%p] idx[%lu] r0[%lu] r1[%lu]\n", __FILE__, __LINE__, p, idx, p->w0, p->w1);
  if (idx < p->w0) p->w0 = idx;
  if (idx > p->w1) p->w1 = idx;
}

static void brisbane_poly_mawr(brisbane_poly_mem* p, size_t idx) {
  return brisbane_poly_muwr(p, idx);
}

#endif /* BRISBANE_INCLUDE_BRISBANE_POLY_H */

