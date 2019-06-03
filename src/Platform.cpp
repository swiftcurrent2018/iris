#include "Platform.h"
#include "Utils.h"
#include "Command.h"
#include "Device.h"
#include "Executor.h"
#include "Kernel.h"
#include "History.h"
#include "Mem.h"
#include "Task.h"
#include <unistd.h>

namespace brisbane {
namespace rt {

char debug_prefix_[256];

Platform::Platform() {
    init_ = false;
    ndevs_ = 0;
    srand(time(NULL));
}

Platform::~Platform() {
    if (!init_) return;
}

int Platform::Init(int* argc, char*** argv) {
    if (init_) return BRISBANE_ERR;
    gethostname(debug_prefix_, 256);
    Utils::Logo(true);
    GetCLPlatforms();
    return BRISBANE_OK;
}

int Platform::GetCLPlatforms() {
    cl_uint num_platforms = 16;
    cl_uint num_devices;

    clerr = clGetPlatformIDs(num_platforms, cl_platforms_, &num_platforms);
    _trace("num_platforms[%u]", num_platforms);
    char platform_vendor[64];
    char platform_name[64];
    for (cl_uint i = 0; i < num_platforms; i++) {
        clerr = clGetPlatformInfo(cl_platforms_[i], CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, NULL);
        clerr = clGetPlatformInfo(cl_platforms_[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        clerr = clGetDeviceIDs(cl_platforms_[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        clerr = clGetDeviceIDs(cl_platforms_[i], CL_DEVICE_TYPE_ALL, num_devices, cl_devices_ + ndevs_, NULL);
        cl_contexts_[i] = clCreateContext(NULL, num_devices, cl_devices_ + ndevs_, NULL, NULL, &clerr);
        for (cl_uint j = 0; j < num_devices; j++) {
            devices_[ndevs_] = new Device(cl_devices_[ndevs_], cl_contexts_[i], ndevs_, i);
            ndevs_++;
        }
    }
}

Device* Platform::AvailableDevice(Task* task, int brs_device) {
    if (brs_device == brisbane_device_default)  return devices_[0];
    if (brs_device == brisbane_device_history)  return GetDeviceHistory(task);
    if (brs_device == brisbane_device_all)      return GetDeviceAll(task);
    if (brs_device == brisbane_device_data)     return GetDeviceData(task);
    if (brs_device == brisbane_device_random)   return GetDeviceRandom(task);
    for (int i = 0; i < ndevs_; i++) {
        Device* dev = devices_[i];
        if (dev->type() == brs_device) return dev;
    }
    return NULL;
}

Device* Platform::GetDeviceAll(Task* task) {
    _check();
    return devices_[0];
}

Device* Platform::GetDeviceHistory(Task* task) {
    Command* cmd = task->cmd_kernel();
    if (!cmd) return devices_[0];
    Kernel* kernel = cmd->kernel();
    History* history = kernel->history();
    Device* dev = history->OptimalDevice(task);
    return dev;
}

Device* Platform::GetDeviceData(Task* task) {
    size_t total_size[16];
    for (int i = 0; i < 16; i++) total_size[i] = 0UL;
    for (int i = 0; i < task->num_cmds(); i++) {
        Command* cmd = task->cmd(i);
        if (cmd->type() == BRISBANE_CMD_KERNEL) {
            Kernel* kernel = cmd->kernel();
            std::map<int, KernelArg*> args = kernel->args();
            for (std::map<int, KernelArg*>::iterator it = args.begin(); it != args.end(); ++it) {
                Mem* mem = it->second->mem;
                if (!mem || !mem->owner()) continue;
                total_size[mem->owner()->dev_no()] += mem->size();
            }
        } else if (cmd->type() == BRISBANE_CMD_H2D || cmd->type() == BRISBANE_CMD_D2H) {
            Mem* mem = cmd->mem();
            if (!mem || !mem->owner()) continue;
            total_size[mem->owner()->dev_no()] += mem->size();
        }
    }
    int target_dev = 0;
    size_t max_size = 0;
    for (int i = 0; i < 16; i++) {
        if (total_size[i] > max_size) {
            max_size = total_size[i];
            target_dev = i;
        }
    }
    return devices_[target_dev];
}

Device* Platform::GetDeviceRandom(Task* task) {
    return devices_[rand() % ndevs_];
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
    Task* task = new Task(NULL);
    *brs_task = task->struct_obj();
    return BRISBANE_OK;
}

int Platform::TaskSubCreate(brisbane_task brs_task, brisbane_task* brs_subtask) {
    Task* task = brs_task->class_obj;
    Task* subtask = new Task(task);
    *brs_subtask = subtask->struct_obj();
    return BRISBANE_OK;
}

int Platform::TaskKernel(brisbane_task brs_task, brisbane_kernel brs_kernel, int dim, size_t* off, size_t* ndr) {
    Task* task = brs_task->class_obj;
    Kernel* kernel = brs_kernel->class_obj;
    Command* cmd = Command::CreateKernel(kernel, dim, off, ndr);
    task->AddCommand(cmd);
    return BRISBANE_OK;
}

int Platform::TaskH2D(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host) {
    Task* task = brs_task->class_obj;
    Mem* mem = brs_mem->class_obj;
    Command* cmd = Command::CreateH2D(mem, off, size, host);
    task->AddCommand(cmd);
    return BRISBANE_OK;
}

int Platform::TaskD2H(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host) {
    Task* task = brs_task->class_obj;
    Mem* mem = brs_mem->class_obj;
    Command* cmd = Command::CreateD2H(mem, off, size, host);
    task->AddCommand(cmd);
    return BRISBANE_OK;
}

int Platform::TaskPresent(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size) {
    Task* task = brs_task->class_obj;
    Mem* mem = brs_mem->class_obj;
    Command* cmd = Command::CreatePresent(mem, off, size);
    task->AddCommand(cmd);
    return BRISBANE_OK;
}

int Platform::TaskSubmit(brisbane_task brs_task, int brs_device, char* opt, bool wait) {
    Task* task = brs_task->class_obj;
    task->Submit(brs_device);
    if (wait) task->Wait();
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

void Platform::ExecuteTask(Task* task) {
    executor_->Execute(task);
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
