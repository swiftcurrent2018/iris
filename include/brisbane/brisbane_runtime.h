#ifndef BRINSBANE_INCLUDE_BRISBANE_BRISBANE_RUNTIME_H
#define BRINSBANE_INCLUDE_BRISBANE_BRISBANE_RUNTIME_H

#include <brisbane/brisbane_errno.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BRISBANE_MAX_NPLATFORMS     32
#define BRISBANE_MAX_NDEVS          (1 << 5) - 1

#define brisbane_default            (1 << 5)
#define brisbane_cpu                (1 << 6)
#define brisbane_nvidia             (1 << 7)
#define brisbane_amd                (1 << 8)
#define brisbane_gpu                (brisbane_nvidia | brisbane_amd)
#define brisbane_phi                (1 << 9)
#define brisbane_fpga               (1 << 10)
#define brisbane_data               (1 << 11)
#define brisbane_profile            (1 << 12)
#define brisbane_random             (1 << 13)
#define brisbane_any                (1 << 14)
#define brisbane_all                (1 << 15)
#define brisbane_custom             (1 << 30)

#define brisbane_r                  (1 << 0)
#define brisbane_w                  (1 << 1)
#define brisbane_rw                 (brisbane_r | brisbane_w)

#define brisbane_int                (1 << 0)
#define brisbane_long               (1 << 1)
#define brisbane_float              (1 << 2)
#define brisbane_double             (1 << 3)

#define brisbane_normal             (1 << 10)
#define brisbane_reduction          (1 << 11)
#define brisbane_sum                ((1 << 12) | brisbane_reduction)
#define brisbane_max                ((1 << 13) | brisbane_reduction)
#define brisbane_min                ((1 << 14) | brisbane_reduction)

#define brisbane_platform           0x1001
#define brisbane_vendor             0x1002
#define brisbane_name               0x1003
#define brisbane_type               0x1004

typedef struct _brisbane_task*      brisbane_task;
typedef struct _brisbane_mem*       brisbane_mem;
typedef struct _brisbane_kernel*    brisbane_kernel;

extern int brisbane_init(int* argc, char*** argv, int sync);
extern int brisbane_finalize();
extern int brisbane_synchronize();

extern int brisbane_platform_count(int* nplatforms);
extern int brisbane_platform_info(int platform, int param, void* value, size_t* size);

extern int brisbane_device_count(int* ndevs);
extern int brisbane_device_info(int device, int param, void* value, size_t* size);
extern int brisbane_device_set_default(int device);
extern int brisbane_device_get_default(int* device);

extern int brisbane_policy_register(const char* lib, const char* name);

extern int brisbane_kernel_create(const char* name, brisbane_kernel* kernel);
extern int brisbane_kernel_get(const char* name, brisbane_kernel* kernel);
extern int brisbane_kernel_setarg(brisbane_kernel kernel, int idx, size_t size, void* value);
extern int brisbane_kernel_setmem(brisbane_kernel kernel, int idx, brisbane_mem mem, int mode);
extern int brisbane_kernel_setmap(brisbane_kernel kernel, int idx, void* host, int mode);
extern int brisbane_kernel_release(brisbane_kernel kernel);

extern int brisbane_task_create(brisbane_task* task);
extern int brisbane_task_create_name(const char* name, brisbane_task* task);
extern int brisbane_task_depend(brisbane_task task, int ntasks, brisbane_task* tasks);
extern int brisbane_task_h2d(brisbane_task task, brisbane_mem mem, size_t off, size_t size, void* host);
extern int brisbane_task_d2h(brisbane_task task, brisbane_mem mem, size_t off, size_t size, void* host);
extern int brisbane_task_h2d_full(brisbane_task task, brisbane_mem mem, void* host);
extern int brisbane_task_d2h_full(brisbane_task task, brisbane_mem mem, void* host);
extern int brisbane_task_mapto(brisbane_task task, void* host, size_t size);
extern int brisbane_task_mapfrom(brisbane_task task, void* host, size_t size);
extern int brisbane_task_kernel(brisbane_task task, brisbane_kernel kernel, int dim, size_t* off, size_t* ndr);
extern int brisbane_task_submit(brisbane_task task, int device, const char* opt, int sync);
extern int brisbane_task_wait(brisbane_task task);
extern int brisbane_task_wait_all(int ntasks, brisbane_task* tasks);
extern int brisbane_task_add_subtask(brisbane_task task, brisbane_task subtask);
extern int brisbane_task_release(brisbane_task task);
extern int brisbane_task_release_mem(brisbane_task task, brisbane_mem mem);

extern int brisbane_mem_create(size_t size, brisbane_mem* mem);
extern int brisbane_mem_map(void* host, size_t size);
extern int brisbane_mem_unmap(void* host);
extern int brisbane_mem_reduce(brisbane_mem mem, int mode, int type);
extern int brisbane_mem_release(brisbane_mem mem);

extern int brisbane_timer_now(double* time);

#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif /* BRINSBANE_INCLUDE_BRISBANE_BRISBANE_RUNTIME_H */

