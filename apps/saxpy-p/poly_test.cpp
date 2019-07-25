#include <stdio.h>
#include <dlfcn.h>

typedef struct {
    size_t typesz;
    size_t offset;
    size_t length;
    int    rwmode;
    bool   enable;
} brisbane_poly_range;

int main() {
    int err;

    void* handle;
    int (*brisbane_poly_kernel)(const char* name);
    int (*brisbane_poly_setarg)(int idx, size_t size, void* value);
    int (*brisbane_poly_launch)(int dim, size_t* off, size_t* ndr);
    int (*brisbane_poly_getmem)(int idx, brisbane_poly_range* range);

    handle = dlopen("./libbrisbane_poly_kernels.so", RTLD_LAZY);
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
    if (!err) printf("[%s:%d] error[%d]\n", __FILE__, __LINE__, err);

    float A = 10;
    err = brisbane_poly_setarg(1, sizeof(A), &A);
    if (!err) printf("[%s:%d] error[%d]\n", __FILE__, __LINE__, err);

    size_t off[] = { 40 };
    size_t len[] = { 1000 };
    err = brisbane_poly_launch(1, off, len);
    if (!err) printf("[%s:%d] error[%d]\n", __FILE__, __LINE__, err);

    brisbane_poly_range Z;
    brisbane_poly_range X;
    brisbane_poly_range Y;
    err = brisbane_poly_getmem(0, &Z);
    if (!err) printf("[%s:%d] error[%d]\n", __FILE__, __LINE__, err);
    err = brisbane_poly_getmem(2, &X);
    if (!err) printf("[%s:%d] error[%d]\n", __FILE__, __LINE__, err);
    err = brisbane_poly_getmem(3, &Y);
    if (!err) printf("[%s:%d] error[%d]\n", __FILE__, __LINE__, err);

    printf("[%s:%d] Z typesz[%lu] offset[%lu] length[%lu] rwmode[%x]\n", __FILE__, __LINE__, Z.typesz, Z.offset, Z.length, Z.rwmode);
    printf("[%s:%d] X typesz[%lu] offset[%lu] length[%lu] rwmode[%x]\n", __FILE__, __LINE__, X.typesz, X.offset, X.length, X.rwmode);
    printf("[%s:%d] Y typesz[%lu] offset[%lu] length[%lu] rwmode[%x]\n", __FILE__, __LINE__, Y.typesz, Y.offset, Y.length, Y.rwmode);

    err = dlclose(handle);
    if (err != 0) printf("[%s:%d] error[%s]\n", __FILE__, __LINE__, dlerror());

    return 0;
}
