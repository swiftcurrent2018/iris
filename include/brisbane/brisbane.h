#ifndef BRINSBANE_INCLUDE_BRISBANE_BRINSBANE_H
#define BRINSBANE_INCLUDE_BRISBANE_BRINSBANE_H

#include <stddef.h>

#define BRISBANE_OK             0
#define BRISBANE_ERR            -1

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _brisbane_mem*       brisbane_mem;
typedef struct _brisbane_kernel*    brisbane_kernel;

#define brisbane_device_default     (1 << 0)
#define brisbane_device_cpu         (1 << 1)
#define brisbane_device_gpu         (1 << 2)
#define brisbane_device_phi         (1 << 3)
#define brisbane_device_fpga        (1 << 4)

extern int brisbane_init(int* argc, char*** argv);
extern int brisbane_finalize();

extern int brisbane_region_begin(int device_type);

extern int brisbane_mem_create(size_t size, brisbane_mem* mem);
extern int brisbane_mem_h2d(brisbane_mem mem, size_t off, size_t size, void* host);

#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif /* BRINSBANE_INCLUDE_BRISBANE_BRINSBANE_H */
