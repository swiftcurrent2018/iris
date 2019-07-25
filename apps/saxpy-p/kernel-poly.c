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
    brisbane_poly_mem Z;
    float A;
    brisbane_poly_mem X;
    brisbane_poly_mem Y;
} brisbane_poly_saxpy_args;

brisbane_poly_saxpy_args saxpy_args;

int brisbane_poly_init() {
    printf("[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
    pthread_mutex_init(&brisbane_poly_mutex, NULL);
}

int brisbane_poly_finalize() {
    printf("[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
    pthread_mutex_destroy(&brisbane_poly_mutex);
}

int brisbane_poly_lock() {
    printf("[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
    pthread_mutex_lock(&brisbane_poly_mutex);
}

int brisbane_poly_unlock() {
    printf("[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
    pthread_mutex_unlock(&brisbane_poly_mutex);
}

int brisbane_poly_saxpy_setarg(int idx, size_t size, void* value) {
    switch (idx) {
        case 1: memcpy(&saxpy_args.A, value, size); break;
        default: return BRISBANE_ERR;
    }

    return BRISBANE_OK;
}

int brisbane_poly_saxpy_launch(int dim, size_t* off, size_t* ndr) {
    saxpy_args.Z.enable = true;
    saxpy_args.Z.typesz = sizeof(float);
    saxpy_args.Z.offset = off[0];
    saxpy_args.Z.length = ndr[0];
    saxpy_args.Z.rwmode = brisbane_w;

    saxpy_args.X.enable = true;
    saxpy_args.X.typesz = sizeof(float);
    saxpy_args.X.offset = off[0];
    saxpy_args.X.length = ndr[0];
    saxpy_args.X.rwmode = brisbane_r;

    saxpy_args.Y.enable = true;
    saxpy_args.Y.typesz = sizeof(float);
    saxpy_args.Y.offset = off[0];
    saxpy_args.Y.length = ndr[0];
    saxpy_args.Y.rwmode = brisbane_r;

    return BRISBANE_OK;
}

int brisbane_poly_saxpy_getmem(int idx, brisbane_poly_mem* range) {
    switch (idx) {
        case 0: memcpy(range, &saxpy_args.Z, sizeof(brisbane_poly_mem));  break;
        case 2: memcpy(range, &saxpy_args.X, sizeof(brisbane_poly_mem));  break;
        case 3: memcpy(range, &saxpy_args.Y, sizeof(brisbane_poly_mem));  break;
        default: return BRISBANE_ERR;
    }
    return BRISBANE_OK;
}

int brisbane_poly_kernel(const char* name) {
    if (strcmp(name, "saxpy") == 0) {
        brisbane_poly_kernel_idx = 0;
        return BRISBANE_OK;
    }
    return BRISBANE_ERR;
}

int brisbane_poly_setarg(int idx, size_t size, void* value) {
    switch (brisbane_poly_kernel_idx) {
        case 0: return brisbane_poly_saxpy_setarg(idx, size, value);
    }
    return BRISBANE_ERR;
}

int brisbane_poly_launch(int dim, size_t* off, size_t* ndr) {
    switch (brisbane_poly_kernel_idx) {
        case 0: return brisbane_poly_saxpy_launch(dim, off, ndr);
    }
    return BRISBANE_ERR;
}

int brisbane_poly_getmem(int idx, brisbane_poly_mem* range) {
    switch (brisbane_poly_kernel_idx) {
        case 0: return brisbane_poly_saxpy_getmem(idx, range);
    }
    return BRISBANE_ERR;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif
