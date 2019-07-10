#include "Platform.h"
#include "Utils.h"
#include "Command.h"
#include "Device.h"
#include "History.h"
#include "Kernel.h"
#include "Mem.h"
#include "Scheduler.h"
#include "Task.h"
#include "Timer.h"
#include <unistd.h>
#include <algorithm>

namespace brisbane {
namespace rt {

char brisbane_log_prefix_[256];

Platform::Platform() {
    init_ = false;
    ndevs_ = 0;
}

Platform::~Platform() {
    if (!init_) return;
    
    if (scheduler_) delete scheduler_;
    if (timer_) delete timer_;
}

int Platform::Init(int* argc, char*** argv) {
    if (init_) return BRISBANE_ERR;
    gethostname(brisbane_log_prefix_, 256);
    Utils::Logo(true);

    timer_ = new Timer();
    timer_->Start(1);

    GetCLPlatforms();

    scheduler_ = new Scheduler(this);
    scheduler_->Start();

    init_ = true;

    return BRISBANE_OK;
}

int Platform::GetCLPlatforms() {
    cl_uint num_platforms = BRISBANE_MAX_NDEVS;
    cl_uint num_devices;

    clerr_ = clGetPlatformIDs(num_platforms, cl_platforms_, &num_platforms);
    _trace("num_platforms[%u]", num_platforms);
    char platform_vendor[64];
    char platform_name[64];
    for (cl_uint i = 0; i < num_platforms; i++) {
        clerr_ = clGetPlatformInfo(cl_platforms_[i], CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, NULL);
        clerr_ = clGetPlatformInfo(cl_platforms_[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        clerr_ = clGetDeviceIDs(cl_platforms_[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        clerr_ = clGetDeviceIDs(cl_platforms_[i], CL_DEVICE_TYPE_ALL, num_devices, cl_devices_ + ndevs_, NULL);
        cl_contexts_[i] = clCreateContext(NULL, num_devices, cl_devices_ + ndevs_, NULL, NULL, &clerr_);
        _clerror(clerr_);
        for (cl_uint j = 0; j < num_devices; j++) {
            devices_[ndevs_] = new Device(cl_devices_[ndevs_], cl_contexts_[i], ndevs_, i);
            ndevs_++;
        }
    }
    return BRISBANE_OK;
}

int Platform::InfoNumDevices(int* ndevs) {
    *ndevs = ndevs_;
    return BRISBANE_OK;
}

int Platform::KernelCreate(const char* name, brisbane_kernel* brs_kernel) {
    for (std::set<Kernel*>::iterator it = kernels_.begin(); it != kernels_.end(); ++it) {
        Kernel* kernel = *it;
        if (strcmp(kernel->name(), name) == 0) {
            *brs_kernel = kernel->struct_obj();
            return BRISBANE_OK;
        }
    }
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

int Platform::KernelSetMem(brisbane_kernel brs_kernel, int idx, brisbane_mem brs_mem, int mode) {
    Kernel* kernel = brs_kernel->class_obj;
    Mem* mem = brs_mem->class_obj;
    kernel->SetMem(idx, mem, mode);
    return BRISBANE_OK;
}

int Platform::KernelRelease(brisbane_kernel brs_kernel) {
    return BRISBANE_OK;
}

int Platform::TaskCreate(brisbane_task* brs_task) {
    Task* task = new Task(this);
    *brs_task = task->struct_obj();
    return BRISBANE_OK;
}

int Platform::TaskKernel(brisbane_task brs_task, brisbane_kernel brs_kernel, int dim, size_t* off, size_t* ndr) {
    Task* task = brs_task->class_obj;
    Kernel* kernel = brs_kernel->class_obj;
    Command* cmd = Command::CreateKernel(task, kernel, dim, off, ndr);
    task->AddCommand(cmd);
    return BRISBANE_OK;
}

int Platform::TaskH2D(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host) {
    Task* task = brs_task->class_obj;
    Mem* mem = brs_mem->class_obj;
    Command* cmd = Command::CreateH2D(task, mem, off, size, host);
    task->AddCommand(cmd);
    return BRISBANE_OK;
}

int Platform::TaskD2H(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host) {
    Task* task = brs_task->class_obj;
    Mem* mem = brs_mem->class_obj;
    Command* cmd = Command::CreateD2H(task, mem, off, size, host);
    task->AddCommand(cmd);
    return BRISBANE_OK;
}

int Platform::TaskH2DFull(brisbane_task brs_task, brisbane_mem brs_mem, void* host) {
    return TaskH2D(brs_task, brs_mem, 0ULL, brs_mem->class_obj->size(), host);
}

int Platform::TaskD2HFull(brisbane_task brs_task, brisbane_mem brs_mem, void* host) {
    return TaskD2H(brs_task, brs_mem, 0ULL, brs_mem->class_obj->size(), host);
}

int Platform::TaskPresent(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host) {
    Task* task = brs_task->class_obj;
    Mem* mem = brs_mem->class_obj;
    Command* cmd = Command::CreatePresent(task, mem, off, size, host);
    task->AddCommand(cmd);
    return BRISBANE_OK;
}

int Platform::TaskSubmit(brisbane_task brs_task, int brs_device, char* opt, bool wait) {
    Task* task = brs_task->class_obj;
    task->set_brs_device(brs_device);
    scheduler_->Enqueue(task);
    if (wait) task->Wait();
    return BRISBANE_OK;
}

int Platform::TaskWait(brisbane_task brs_task) {
    Task* task = brs_task->class_obj;
    task->Wait();
    return BRISBANE_OK;
}

int Platform::TaskAddSubtask(brisbane_task brs_task, brisbane_task brs_subtask) {
    Task* task = brs_task->class_obj;
    Task* subtask = brs_subtask->class_obj;
    task->AddSubtask(subtask);
    return BRISBANE_OK;
}

int Platform::TaskRelease(brisbane_task brs_task) {
    Task* task = brs_task->class_obj;
    delete task;
}

int Platform::TaskReleaseMem(brisbane_task brs_task, brisbane_mem brs_mem) {
    Task* task = brs_task->class_obj;
    Mem* mem = brs_mem->class_obj;
    Command* cmd = Command::CreateReleaseMem(task, mem);
    task->AddCommand(cmd);
    return BRISBANE_OK;
}

int Platform::MemCreate(size_t size, brisbane_mem* brs_mem) {
    Mem* mem = new Mem(size, this);
    *brs_mem = mem->struct_obj();
    mems_.insert(mem);
    return BRISBANE_OK;
}

int Platform::MemReduce(brisbane_mem brs_mem, int mode, int type) {
    Mem* mem = brs_mem->class_obj;
    mem->Reduce(mode, type);
    return BRISBANE_OK;
}

int Platform::MemRelease(brisbane_mem brs_mem) {
    Mem* mem = brs_mem->class_obj;
    delete mem;
    return BRISBANE_OK;
}

int Platform::TimerNow(double* time) {
    *time = timer_->Now();
    return BRISBANE_OK;
}

int Platform::ShowKernelHistory() {
    for (std::set<Kernel*>::iterator it = kernels_.begin(); it != kernels_.end(); ++it) {
        Kernel* kernel = *it;
        History* history = kernel->history();
        _info("kernel[%s] k[%lf][%lu] h2d[%lf][%lu] d2h[%lf][%lu]", kernel->name(), history->t_kernel(), history->c_kernel(), history->t_h2d(), history->c_h2d(), history->t_d2h(), history->c_d2h());
    }
    return BRISBANE_OK;
}

Platform* Platform::singleton_ = NULL;

Platform* Platform::GetPlatform() {
    if (singleton_ == NULL) singleton_ = new Platform();
    return singleton_;
}

int Platform::Finalize() {
    singleton_->ShowKernelHistory();
    double total = singleton_->timer()->Stop(1);
    if (singleton_ == NULL) return BRISBANE_ERR;
    if (singleton_) delete singleton_;
    singleton_ = NULL;
    _info("total execution time [%lf] sec", total);
    return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */
