#include "Platform.h"
#include "Utils.h"
#include "Command.h"
#include "Device.h"
#include "Kernel.h"
#include "Mem.h"
#include "Task.h"
#include <unistd.h>

namespace brisbane {
namespace rt {

char debug_prefix_[256];

Platform::Platform() {
    init_ = false;
    num_devices_ = 0;
}

Platform::~Platform() {
    if (!init_) return;
}

int Platform::Init(int* argc, char*** argv) {
    if (init_) return BRISBANE_ERR;
    gethostname(debug_prefix_, 256);
    Utils::Logo();
    GetCLPlatforms();
    return BRISBANE_OK;
}

int Platform::GetCLPlatforms() {
    cl_uint num_platforms;
    cl_uint num_devices;

    clerr = clGetPlatformIDs(0, NULL, &num_platforms);
    _trace("num_platforms[%u]", num_platforms);
    clerr = clGetPlatformIDs(num_platforms, cl_platforms_, NULL);
    char platform_vendor[64];
    char platform_name[64];
    for (cl_uint i = 0; i < num_platforms; i++) {
        clerr = clGetPlatformInfo(cl_platforms_[i], CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, NULL);
        clerr = clGetPlatformInfo(cl_platforms_[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        clerr = clGetDeviceIDs(cl_platforms_[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        clerr = clGetDeviceIDs(cl_platforms_[i], CL_DEVICE_TYPE_ALL, num_devices, cl_devices_ + num_devices_, NULL);
        cl_contexts_[i] = clCreateContext(NULL, num_devices, cl_devices_ + num_devices_, NULL, NULL, &clerr);
        for (cl_uint j = 0; j < num_devices; j++) {
            devices_[num_devices_] = new Device(cl_devices_[num_devices_], cl_contexts_[i], num_devices_, i);
            num_devices_++;
        }
    }
}

Device* Platform::AvailableDevice(int brs_device) {
    for (int i = 0; i < num_devices_; i++) {
        Device* dev = devices_[i];
        if (dev->type() == brs_device) return dev;
    }
    return NULL;
}

int Platform::KernelCreate(const char* name, brisbane_kernel* brs_kernel) {
    Kernel* kernel = new Kernel(name, this);
    *brs_kernel = kernel->struct_obj();
    kernels_.insert(kernel);
    return BRISBANE_OK;
}

int Platform::KernelSetArg(brisbane_kernel brs_kernel, int idx, size_t arg_size, void* arg_value) {
    Kernel* kernel = brs_kernel->class_obj;
    kernel->SetArg(idx, arg_size, arg_value);
    return BRISBANE_OK;
}

int Platform::KernelRelease(brisbane_kernel brs_kernel) {
    return BRISBANE_OK;
}

int Platform::TaskCreate(brisbane_task* brs_task) {
    Task* task = new Task();
    *brs_task = task->struct_obj();
    return BRISBANE_OK;
}

int Platform::TaskH2D(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host) {
    Task* task = brs_task->class_obj;
    Mem* mem = brs_mem->class_obj;
    Command* cmd = Command::CreateH2D(mem, off, size, host);
    task->Add(cmd);
    return BRISBANE_OK;
}

int Platform::TaskD2H(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host) {
    Task* task = brs_task->class_obj;
    Mem* mem = brs_mem->class_obj;
    Command* cmd = Command::CreateD2H(mem, off, size, host);
    task->Add(cmd);
    return BRISBANE_OK;
}

int Platform::TaskKernel(brisbane_task brs_task, brisbane_kernel brs_kernel, int dim, size_t* ndr) {
    Task* task = brs_task->class_obj;
    Kernel* kernel = brs_kernel->class_obj;
    Command* cmd = Command::CreateKernel(kernel, dim, ndr);
    task->Add(cmd);
    return BRISBANE_OK;
}

int Platform::TaskSubmit(brisbane_task brs_task, int brs_device) {
    Task* task = brs_task->class_obj;
    task->Submit(brs_device);
    return BRISBANE_OK;
}

int Platform::TaskWait(brisbane_task brs_task) {
    Task* task = brs_task->class_obj;
    task->Wait();
    return BRISBANE_OK;
}

int Platform::TaskRelease(brisbane_task brs_task) {
    return BRISBANE_OK;
}

int Platform::MemCreate(size_t size, brisbane_mem* brs_mem) {
    Mem* mem = new Mem(size);
    *brs_mem = mem->struct_obj();
    mems_.insert(mem);
    return BRISBANE_OK;
}

int Platform::MemRelease(brisbane_mem brs_mem) {
    return BRISBANE_OK;
}

Mem* Platform::GetMemFromPtr(void* ptr) {
    for (std::set<Mem*>::iterator it = mems_.begin(); it != mems_.end(); ++it) {
        Mem* mem = *it;
        brisbane_mem brs_mem = *((brisbane_mem*) ptr);
        if (brs_mem == mem->struct_obj()) return mem;
    }
    return NULL;
}

Platform* Platform::singleton_ = NULL;

Platform* Platform::GetPlatform() {
    if (singleton_ == NULL) singleton_ = new Platform();
    return singleton_;
}

int Platform::Finalize() {
    if (singleton_ == NULL) return BRISBANE_ERR;
    if (singleton_) delete singleton_;
    singleton_ = NULL;
    return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */
