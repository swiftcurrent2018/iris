#ifndef BRINSBANE_INCLUDE_BRISBANE_BRISBANE_H
#define BRINSBANE_INCLUDE_BRISBANE_BRISBANE_H

#include <stddef.h>

#define BRISBANE_OK             0
#define BRISBANE_ERR            -1

#ifdef __cplusplus
extern "C" {
#endif

#define brisbane_default            (1 << 0)
#define brisbane_cpu                (1 << 1)
#define brisbane_gpu                (1 << 2)
#define brisbane_phi                (1 << 3)
#define brisbane_fpga               (1 << 4)
#define brisbane_data               (1 << 5)
#define brisbane_history            (1 << 6)
#define brisbane_random             (1 << 7)
#define brisbane_any                (brisbane_cpu | brisbane_gpu | brisbane_phi | brisbane_fpga)

#define brisbane_device_default     (1 << 0)
#define brisbane_device_cpu         (1 << 1)
#define brisbane_device_gpu         (1 << 2)
#define brisbane_device_phi         (1 << 3)
#define brisbane_device_fpga        (1 << 4)
#define brisbane_device_data        (1 << 5)
#define brisbane_device_history     (1 << 6)
#define brisbane_device_random      (1 << 7)
#define brisbane_device_any         (brisbane_device_cpu | brisbane_device_gpu | brisbane_device_phi | brisbane_device_fpga)

#define brisbane_rd                 (1 << 0)
#define brisbane_wr                 (1 << 1)
#define brisbane_rw                 (brisbane_rd | brisbane_wr)
#define brisbane_rdwr               (brisbane_rd | brisbane_wr)

#define brisbane_int                (1 << 0)
#define brisbane_long               (1 << 1)
#define brisbane_float              (1 << 2)
#define brisbane_double             (1 << 3)

#define brisbane_normal             (1 << 10)
#define brisbane_reduction          (1 << 11)
#define brisbane_sum                ((1 << 12) | brisbane_reduction)
#define brisbane_max                ((1 << 13) | brisbane_reduction)
#define brisbane_min                ((1 << 14) | brisbane_reduction)


typedef struct _brisbane_task*      brisbane_task;
typedef struct _brisbane_mem*       brisbane_mem;
typedef struct _brisbane_kernel*    brisbane_kernel;

extern int brisbane_init(int* argc, char*** argv);
extern int brisbane_finalize();

extern int brisbane_info_ndevs(int* ndevs);

extern int brisbane_kernel_create(const char* name, brisbane_kernel* kernel);
extern int brisbane_kernel_setarg(brisbane_kernel kernel, int idx, size_t arg_size, void* arg_value);
extern int brisbane_kernel_setmem(brisbane_kernel kernel, int idx, brisbane_mem mem, int mode);
extern int brisbane_kernel_release(brisbane_kernel kernel);

extern int brisbane_task_create(brisbane_task* task);
extern int brisbane_task_kernel(brisbane_task task, brisbane_kernel kernel, int dim, size_t* off, size_t* ndr);
extern int brisbane_task_h2d(brisbane_task task, brisbane_mem mem, size_t off, size_t size, void* host);
extern int brisbane_task_d2h(brisbane_task task, brisbane_mem mem, size_t off, size_t size, void* host);
extern int brisbane_task_h2d_full(brisbane_task task, brisbane_mem mem, void* host);
extern int brisbane_task_d2h_full(brisbane_task task, brisbane_mem mem, void* host);
extern int brisbane_task_present(brisbane_task task, brisbane_mem mem, size_t off, size_t size, void* host);
extern int brisbane_task_submit(brisbane_task task, int device, char* opt, bool wait);
extern int brisbane_task_wait(brisbane_task task);
extern int brisbane_task_add_subtask(brisbane_task task, brisbane_task subtask);
extern int brisbane_task_release(brisbane_task task);
extern int brisbane_task_release_mem(brisbane_task task, brisbane_mem mem);

extern int brisbane_mem_create(size_t size, brisbane_mem* mem);
extern int brisbane_mem_reduce(brisbane_mem mem, int mode, int type);
extern int brisbane_mem_release(brisbane_mem mem);

extern int brisbane_timer_now(double* time);

#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif /* BRINSBANE_INCLUDE_BRISBANE_BRISBANE_H */
