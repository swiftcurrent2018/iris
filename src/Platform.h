#ifndef BRISBANE_RT_SRC_PLATFORM_H
#define BRISBANE_RT_SRC_PLATFORM_H

#define CL_TARGET_OPENCL_VERSION 120

#include <brisbane/brisbane.h>
#include <CL/cl.h>
#include <stddef.h>
#include <set>
#include "Define.h"
#include "Debug.h"

namespace brisbane {
namespace rt {

class Device;
class Kernel;
class Mem;
class Scheduler;
class Task;
class Timer;

class Platform {
private:
    Platform();
    ~Platform();

public:
    int Init(int* argc, char*** argv);
    int GetCLPlatforms();

    int KernelCreate(const char* name, brisbane_kernel* brs_kernel);
    int KernelSetArg(brisbane_kernel kernel, int idx, size_t arg_size, void* arg_value);
    int KernelSetMem(brisbane_kernel kernel, int idx, brisbane_mem mem, int mode);
    int KernelRelease(brisbane_kernel kernel);

    int TaskCreate(brisbane_task* brs_task);
    int TaskKernel(brisbane_task brs_task, brisbane_kernel brs_kernel, int dim, size_t* off, size_t* ndr);
    int TaskH2D(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host);
    int TaskD2H(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host);
    int TaskPresent(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host);
    int TaskSubmit(brisbane_task brs_task, int brs_device, char* opt, bool wait);
    int TaskWait(brisbane_task brs_task);
    int TaskAddSubtask(brisbane_task brs_task, brisbane_task brs_subtask);
    int TaskRelease(brisbane_task brs_task);

    int MemCreate(size_t size, brisbane_mem* brs_mem);
    int MemRelease(brisbane_mem brs_mem);

    int TimerNow(double* time);

    int ndevs() { return ndevs_; }
    Device** devices() { return devices_; }
    Device* device(int dev_no) { return devices_[dev_no]; }
    Scheduler* scheduler() { return scheduler_; }
    Timer* timer() { return timer_; }

public:
    static Platform* GetPlatform();
    static int Finalize();

private:
    bool init_;

    Device* devices_[BRISBANE_MAX_NDEVS];
    int ndevs_;

    cl_platform_id cl_platforms_[BRISBANE_MAX_NDEVS];
    cl_context cl_contexts_[BRISBANE_MAX_NDEVS];
    cl_device_id cl_devices_[BRISBANE_MAX_NDEVS];
    cl_int clerr_;

    std::set<Kernel*> kernels_;
    std::set<Mem*> mems_;

    Scheduler* scheduler_;
    Timer* timer_;

private:
    static Platform* singleton_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_PLATFORM_H */
