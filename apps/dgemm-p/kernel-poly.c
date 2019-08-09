#include <string.h>
#include <stdio.h>
#include <brisbane/brisbane.h>
#include <brisbane/brisbane_internal.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

pthread_mutex_t brisbane_poly_mutex;
int brisbane_poly_kernel_idx;

typedef struct {
  brisbane_poly_mem C;
  brisbane_poly_mem A;
  brisbane_poly_mem B;
  int SIZE;
} brisbane_poly_ijk_args;

brisbane_poly_ijk_args ijk_args;

int brisbane_poly_init() {
  printf("[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
  pthread_mutex_init(&brisbane_poly_mutex, NULL);
  return BRISBANE_OK;
}

int brisbane_poly_finalize() {
  printf("[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
  pthread_mutex_destroy(&brisbane_poly_mutex);
  return BRISBANE_OK;
}

int brisbane_poly_lock() {
  printf("[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
  pthread_mutex_lock(&brisbane_poly_mutex);
  return BRISBANE_OK;
}

int brisbane_poly_unlock() {
  printf("[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
  pthread_mutex_unlock(&brisbane_poly_mutex);
  return BRISBANE_OK;
}

int brisbane_poly_ijk_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    case 3: memcpy(&ijk_args.SIZE, value, size); break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

int brisbane_poly_ijk_launch(int dim, size_t* off, size_t* ndr) {
  int SIZE = ijk_args.SIZE;

  ijk_args.C.typesz = sizeof(double);
  ijk_args.C.poly_r = true;
  ijk_args.C.off_r  = 0;
  ijk_args.C.len_r  = 0;
  ijk_args.C.poly_w = true;
  ijk_args.C.off_w  = off[1] * SIZE;
  ijk_args.C.len_w  = ndr[1] * SIZE;

  ijk_args.A.typesz = sizeof(double);
  ijk_args.A.poly_r = true;
  ijk_args.A.off_r  = off[0];
  ijk_args.A.len_r  = ndr[0];
  ijk_args.A.poly_w = true;
  ijk_args.A.off_w  = off[1] * SIZE;
  ijk_args.A.len_w  = ndr[1] * SIZE;

  ijk_args.B.typesz = sizeof(double);
  ijk_args.B.poly_r = true;
  ijk_args.B.off_r  = 0;
  ijk_args.B.len_r  = SIZE * SIZE;
  ijk_args.B.poly_w = true;
  ijk_args.B.off_w  = 0;
  ijk_args.B.len_w  = 0;

  return BRISBANE_OK;
}

int brisbane_poly_ijk_getmem(int idx, brisbane_poly_mem* range) {
  switch (idx) {
    case 0: memcpy(range, &ijk_args.C, sizeof(brisbane_poly_mem));  break;
    case 1: memcpy(range, &ijk_args.A, sizeof(brisbane_poly_mem));  break;
    case 2: memcpy(range, &ijk_args.B, sizeof(brisbane_poly_mem));  break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

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

int brisbane_poly_launch(int dim, size_t* off, size_t* ndr) {
  int ret = BRISBANE_OK;
  switch (brisbane_poly_kernel_idx) {
    case 0: ret = brisbane_poly_ijk_launch(dim, off, ndr); break;
  }
  brisbane_poly_unlock();
  return ret;
}

int brisbane_poly_getmem(int idx, brisbane_poly_mem* range) {
  switch (brisbane_poly_kernel_idx) {
    case 0: return brisbane_poly_ijk_getmem(idx, range);
  }
  return BRISBANE_ERR;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif
