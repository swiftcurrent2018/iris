#include <brisbane/brisbane.h>
#include "Debug.h"
#include "Platform.h"

using namespace brisbane::rt;

int brisbane_init(int* argc, char*** argv) {
    return Platform::GetPlatform()->Init(argc, argv);
}

int brisbane_finalize() {
    return Platform::Finalize();
}

int brisbane_info_ndevs(int* ndevs) {
    return Platform::GetPlatform()->InfoNumDevices(ndevs);
}

int brisbane_kernel_create(const char* name, brisbane_kernel* kernel) {
    return Platform::GetPlatform()->KernelCreate(name, kernel);
}

int brisbane_kernel_setarg(brisbane_kernel kernel, int idx, size_t arg_size, void* arg_value) {
    return Platform::GetPlatform()->KernelSetArg(kernel, idx, arg_size, arg_value);
}

int brisbane_kernel_setmem(brisbane_kernel kernel, int idx, brisbane_mem mem, int mode) {
    return Platform::GetPlatform()->KernelSetMem(kernel, idx, mem, mode);
}

int brisbane_kernel_release(brisbane_kernel kernel) {
    return Platform::GetPlatform()->KernelRelease(kernel);
}

int brisbane_task_create(brisbane_task* task) {
    return Platform::GetPlatform()->TaskCreate(task);
}

int brisbane_task_present(brisbane_task task, brisbane_mem mem, size_t off, size_t size, void* host) {
    return Platform::GetPlatform()->TaskPresent(task, mem, off, size, host);
}

int brisbane_task_h2d(brisbane_task task, brisbane_mem mem, size_t off, size_t size, void* host) {
    return Platform::GetPlatform()->TaskH2D(task, mem, off, size, host);
}

int brisbane_task_d2h(brisbane_task task, brisbane_mem mem, size_t off, size_t size, void* host) {
    return Platform::GetPlatform()->TaskD2H(task, mem, off, size, host);
}

int brisbane_task_kernel(brisbane_task task, brisbane_kernel kernel, int dim, size_t* off, size_t* ndr) {
    return Platform::GetPlatform()->TaskKernel(task, kernel, dim, off, ndr);
}

int brisbane_task_submit(brisbane_task task, int device, char* opt, bool wait) {
    return Platform::GetPlatform()->TaskSubmit(task, device, opt, wait);
}

int brisbane_task_wait(brisbane_task task) {
    return Platform::GetPlatform()->TaskWait(task);
}

int brisbane_task_add_subtask(brisbane_task task, brisbane_task subtask) {
    return Platform::GetPlatform()->TaskAddSubtask(task, subtask);
}

int brisbane_task_release(brisbane_task task) {
    return Platform::GetPlatform()->TaskRelease(task);
}

int brisbane_mem_create(size_t size, brisbane_mem* mem) {
    return Platform::GetPlatform()->MemCreate(size, mem);
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

