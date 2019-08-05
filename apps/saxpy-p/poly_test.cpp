#include <stdio.h>
#include <dlfcn.h>
#include <brisbane/brisbane.h>
#include <brisbane/brisbane_internal.h>

int main() {
  int err;

  void* handle;
  int (*brisbane_poly_kernel)(const char* name);
  int (*brisbane_poly_setarg)(int idx, size_t size, void* value);
  int (*brisbane_poly_launch)(int dim, size_t* off, size_t* ndr);
  int (*brisbane_poly_getmem)(int idx, brisbane_poly_mem* range);

  handle = dlopen("./libbrisbane_poly.so", RTLD_LAZY);
  if (!handle) printf("[%s:%d] error[%s]\n", __FILE__, __LINE__, dlerror());

  *(void**) (&brisbane_poly_kernel) = dlsym(handle, "brisbane_poly_kernel");
  if (!brisbane_poly_kernel) printf("[%s:%d] error\n", __FILE__, __LINE__);
  *(void**) (&brisbane_poly_setarg) = dlsym(handle, "brisbane_poly_setarg");
  if (!brisbane_poly_setarg) printf("[%s:%d] error\n", __FILE__, __LINE__);
  *(void**) (&brisbane_poly_launch) = dlsym(handle, "brisbane_poly_launch");
  if (!brisbane_poly_launch) printf("[%s:%d] error\n", __FILE__, __LINE__);
  *(void**) (&brisbane_poly_getmem) = dlsym(handle, "brisbane_poly_getmem");
  if (!brisbane_poly_getmem) printf("[%s:%d] error\n", __FILE__, __LINE__);

  err = brisbane_poly_kernel("saxpy");
  if (err != BRISBANE_OK) printf("[%s:%d] error[%d]\n", __FILE__, __LINE__, err);

  float A = 10;
  err = brisbane_poly_setarg(1, sizeof(A), &A);
  if (err != BRISBANE_OK) printf("[%s:%d] error[%d]\n", __FILE__, __LINE__, err);

  size_t off[] = { 40 };
  size_t len[] = { 1000 };
  err = brisbane_poly_launch(1, off, len);
  if (err != BRISBANE_OK) printf("[%s:%d] error[%d]\n", __FILE__, __LINE__, err);

  brisbane_poly_mem Z;
  brisbane_poly_mem X;
  brisbane_poly_mem Y;
  err = brisbane_poly_getmem(0, &Z);
  if (err != BRISBANE_OK) printf("[%s:%d] error[%d]\n", __FILE__, __LINE__, err);
  err = brisbane_poly_getmem(2, &X);
  if (err != BRISBANE_OK) printf("[%s:%d] error[%d]\n", __FILE__, __LINE__, err);
  err = brisbane_poly_getmem(3, &Y);
  if (err != BRISBANE_OK) printf("[%s:%d] error[%d]\n", __FILE__, __LINE__, err);

  printf("[%s:%d] Z typesz[%lu] poly[%d.%d] offset[%lu,%lu] length[%lu,%lu]\n", __FILE__, __LINE__, Z.typesz, Z.poly_r, Z.poly_w, Z.off_r, Z.off_w, Z.len_r, Z.len_w);
  printf("[%s:%d] X typesz[%lu] poly[%d.%d] offset[%lu,%lu] length[%lu,%lu]\n", __FILE__, __LINE__, X.typesz, X.poly_r, X.poly_w, X.off_r, X.off_w, X.len_r, X.len_w);
  printf("[%s:%d] Y typesz[%lu] poly[%d.%d] offset[%lu,%lu] length[%lu,%lu]\n", __FILE__, __LINE__, Y.typesz, Y.poly_r, Y.poly_w, Y.off_r, Y.off_w, Y.len_r, Y.len_w);

  err = dlclose(handle);
  if (err != 0) printf("[%s:%d] error[%s]\n", __FILE__, __LINE__, dlerror());

  return 0;
}
