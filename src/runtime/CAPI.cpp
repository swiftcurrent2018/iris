#include <brisbane/brisbane.h>
#include "Debug.h"
#include "Platform.h"

using namespace brisbane::rt;

int brisbane_init(int* argc, char*** argv, int sync) {
  return Platform::GetPlatform()->Init(argc, argv, sync);
}

int brisbane_finalize() {
  return Platform::GetPlatform()->Finalize();
}

int brisbane_synchronize() {
  return Platform::GetPlatform()->Synchronize();
}

int brisbane_platform_count(int* nplatforms) {
  return Platform::GetPlatform()->PlatformCount(nplatforms);
}

int brisbane_platform_info(int platform, int param, void* value, size_t* size) {
  return Platform::GetPlatform()->PlatformInfo(platform, param, value, size);
}

int brisbane_device_count(int* ndevs) {
  return Platform::GetPlatform()->DeviceCount(ndevs);
}

int brisbane_device_info(int device, int param, void* value, size_t* size) {
  return Platform::GetPlatform()->DeviceInfo(device, param, value, size);
}

int brisbane_device_set_default(int device) {
  return Platform::GetPlatform()->DeviceSetDefault(device);
}

int brisbane_device_get_default(int* device) {
  return Platform::GetPlatform()->DeviceGetDefault(device);
}

int brisbane_policy_register(const char* lib, const char* name) {
  return Platform::GetPlatform()->PolicyRegister(lib, name);
}

int brisbane_kernel_create(const char* name, brisbane_kernel* kernel) {
  return Platform::GetPlatform()->KernelCreate(name, kernel);
}

int brisbane_kernel_get(const char* name, brisbane_kernel* kernel) {
  return Platform::GetPlatform()->KernelGet(name, kernel);
}

int brisbane_kernel_setarg(brisbane_kernel kernel, int idx, size_t size, void* value) {
  return Platform::GetPlatform()->KernelSetArg(kernel, idx, size, value);
}

int brisbane_kernel_setmem(brisbane_kernel kernel, int idx, brisbane_mem mem, int mode) {
  return Platform::GetPlatform()->KernelSetMem(kernel, idx, mem, mode);
}

int brisbane_kernel_setmap(brisbane_kernel kernel, int idx, void* host, int mode) {
  return Platform::GetPlatform()->KernelSetMap(kernel, idx, host, mode);
}

int brisbane_kernel_release(brisbane_kernel kernel) {
  return Platform::GetPlatform()->KernelRelease(kernel);
}

int brisbane_task_create(brisbane_task* task) {
  return Platform::GetPlatform()->TaskCreate(NULL, task);
}

int brisbane_task_create_name(const char* name, brisbane_task* task) {
  return Platform::GetPlatform()->TaskCreate(name, task);
}

int brisbane_task_depend(brisbane_task task, int ntasks, brisbane_task* tasks) {
  return Platform::GetPlatform()->TaskDepend(task, ntasks, tasks);
}

int brisbane_task_h2d(brisbane_task task, brisbane_mem mem, size_t off, size_t size, void* host) {
  return Platform::GetPlatform()->TaskH2D(task, mem, off, size, host);
}

int brisbane_task_d2h(brisbane_task task, brisbane_mem mem, size_t off, size_t size, void* host) {
  return Platform::GetPlatform()->TaskD2H(task, mem, off, size, host);
}

int brisbane_task_h2d_full(brisbane_task task, brisbane_mem mem, void* host) {
  return Platform::GetPlatform()->TaskH2DFull(task, mem, host);
}

int brisbane_task_d2h_full(brisbane_task task, brisbane_mem mem, void* host) {
  return Platform::GetPlatform()->TaskD2HFull(task, mem, host);
}

int brisbane_task_mapto(brisbane_task task, void* host, size_t size) {
  return Platform::GetPlatform()->TaskMapTo(task, host, size);
}

int brisbane_task_mapfrom(brisbane_task task, void* host, size_t size) {
  return Platform::GetPlatform()->TaskMapFrom(task, host, size);
}

int brisbane_task_kernel(brisbane_task task, brisbane_kernel kernel, int dim, size_t* off, size_t* ndr) {
  return Platform::GetPlatform()->TaskKernel(task, kernel, dim, off, ndr);
}

int brisbane_task_submit(brisbane_task task, int device, const char* opt, int sync) {
  return Platform::GetPlatform()->TaskSubmit(task, device, opt, sync);
}

int brisbane_task_wait(brisbane_task task) {
  return Platform::GetPlatform()->TaskWait(task);
}

int brisbane_task_wait_all(int ntasks, brisbane_task* tasks) {
  return Platform::GetPlatform()->TaskWaitAll(ntasks, tasks);
}

int brisbane_task_add_subtask(brisbane_task task, brisbane_task subtask) {
  return Platform::GetPlatform()->TaskAddSubtask(task, subtask);
}

int brisbane_task_release(brisbane_task task) {
  return Platform::GetPlatform()->TaskRelease(task);
}

int brisbane_task_release_mem(brisbane_task task, brisbane_mem mem) {
  return Platform::GetPlatform()->TaskReleaseMem(task, mem);
}

int brisbane_mem_create(size_t size, brisbane_mem* mem) {
  return Platform::GetPlatform()->MemCreate(size, mem);
}

int brisbane_mem_map(void* host, size_t size) {
  return Platform::GetPlatform()->MemMap(host, size);
}

int brisbane_mem_unmap(void* host) {
  return Platform::GetPlatform()->MemUnmap(host);
}

int brisbane_mem_reduce(brisbane_mem mem, int mode, int type) {
  return Platform::GetPlatform()->MemReduce(mem, mode, type);
}

int brisbane_mem_release(brisbane_mem mem) {
  return Platform::GetPlatform()->MemRelease(mem);
}

int brisbane_timer_now(double* time) {
  return Platform::GetPlatform()->TimerNow(time);
}

