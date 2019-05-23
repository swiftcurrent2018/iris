#ifndef BRISBANE_RT_SRC_PLATFORM_H
#define BRISBANE_RT_SRC_PLATFORM_H

#include <brisbane/brisbane.h>
#include <CL/cl.h>
#include <stddef.h>
#include <set>
#include "Debug.h"

namespace brisbane {
namespace rt {

class Device;
class Kernel;
class Mem;

class Platform {
private:
    Platform();
    ~Platform();

public:
    int Init(int* argc, char*** argv);
    int GetCLPlatforms();

    Device* AvailableDevice(int brs_device);

    int KernelCreate(const char* name, brisbane_kernel* brs_kernel);
    int KernelSetArg(brisbane_kernel kernel, int idx, size_t arg_size, void* arg_value);
    int KernelRelease(brisbane_kernel kernel);

    int TaskCreate(brisbane_task* brs_task);
    int TaskH2D(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host);
    int TaskD2H(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host);
    int TaskKernel(brisbane_task brs_task, brisbane_kernel brs_kernel, int dim, size_t* ndr);
    int TaskSubmit(brisbane_task brs_task, int brs_device);
    int TaskWait(brisbane_task brs_task);
    int TaskRelease(brisbane_task brs_task);

    int MemCreate(size_t size, brisbane_mem* brs_mem);
    int MemRelease(brisbane_mem brs_mem);

    Mem* GetMemFromPtr(void* ptr);

public:
    static Platform* GetPlatform();
    static int Finalize();

private:
    bool init_;

    Device* devices_[16];
    int num_devices_;

    cl_platform_id cl_platforms_[16];
    cl_context cl_contexts_[16];
    cl_device_id cl_devices_[16];
    cl_int clerr;

    std::set<Kernel*> kernels_;
    std::set<Mem*> mems_;

private:
    static Platform* singleton_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_PLATFORM_H */
