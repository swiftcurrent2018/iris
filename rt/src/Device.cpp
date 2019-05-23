#include "Device.h"
#include "Kernel.h"
#include "Mem.h"
#include "Utils.h"

namespace brisbane {
namespace rt {

Device::Device(cl_device_id cldev, cl_context clctx, int dev_no, int platform_no) {
    cldev_ = cldev;
    clctx_ = clctx;
    dev_no_ = dev_no;
    platform_no_ = platform_no;
    clerr_ = clGetDeviceInfo(cldev_, CL_DEVICE_VENDOR, sizeof(vendor_), vendor_, NULL);
    clerr_ = clGetDeviceInfo(cldev_, CL_DEVICE_NAME, sizeof(name_), name_, NULL);
    clerr_ = clGetDeviceInfo(cldev_, CL_DEVICE_TYPE, sizeof(cltype_), &cltype_, NULL);
    clerr_ = clGetDeviceInfo(cldev_, CL_DEVICE_VERSION, sizeof(version_), version_, NULL);

    if (cltype_ == CL_DEVICE_TYPE_CPU) type_ = brisbane_device_cpu;
    else if (cltype_ == CL_DEVICE_TYPE_GPU) type_ = brisbane_device_gpu;
    else if (cltype_ == CL_DEVICE_TYPE_ACCELERATOR) type_ = brisbane_device_fpga;
    else type_ = brisbane_device_phi;

    _info("vendor[%s] device[%s] type[0x%x] version[%s]", vendor_, name_, type_, version_);

    clcmdq_ = clCreateCommandQueue(clctx_, cldev_, 0, &clerr_);

    BuildProgram();
}

Device::~Device() {
}

void Device::BuildProgram() {
    char path[256];
    memset(path, 0, 256);
    sprintf(path, "kernel-%s.cl",
            type_ == brisbane_device_cpu  ? "cpu"  :
            type_ == brisbane_device_gpu  ? "gpu"  :
            type_ == brisbane_device_fpga ? "fpga" : "default");
    char* src = NULL;
    size_t srclen = 0;
    Utils::ReadFile(path, &src, &srclen);
    clprog_ = clCreateProgramWithSource(clctx_, 1, (const char**) &src, (const size_t*) &srclen, &clerr_);
    if (clerr_ != CL_SUCCESS) _error("clerr[%d]", clerr_);

    clerr_ = clBuildProgram(clprog_, 1, &cldev_, "", NULL, NULL);
    if (clerr_ != CL_SUCCESS) _error("clerr[%d]", clerr_);

    free(src);
}

void Device::Execute(Task* task) {
    for (int i = 0; i < task->num_cmds(); i++) {
        Command* cmd = task->cmd(i);
        switch (cmd->type()) {
            case BRISBANE_CMD_H2D:      ExecuteH2D(cmd);        break;
            case BRISBANE_CMD_D2H:      ExecuteD2H(cmd);        break;
            case BRISBANE_CMD_KERNEL:   ExecuteKernel(cmd);     break;
            default: _error("cmd type[0x%x]", cmd->type());
        }
    }
}

void Device::ExecuteH2D(Command* cmd) {
    Mem* mem = cmd->mem();
    cl_mem clmem = mem->clmem(platform_no_, clctx_);
    size_t off = cmd->off();
    size_t size = cmd->size();
    void* host = cmd->host();

    clerr_ = clEnqueueWriteBuffer(clcmdq_, clmem, CL_TRUE, off, size, host, 0, NULL, NULL);
    if (clerr_ != CL_SUCCESS) _error("clerr[%d]", clerr_);
}

void Device::ExecuteD2H(Command* cmd) {
    Mem* mem = cmd->mem();
    cl_mem clmem = mem->clmem(platform_no_, clctx_);
    size_t off = cmd->off();
    size_t size = cmd->size();
    void* host = cmd->host();

    clerr_ = clEnqueueReadBuffer(clcmdq_, clmem, CL_TRUE, off, size, host, 0, NULL, NULL);
    if (clerr_ != CL_SUCCESS) _error("clerr[%d]", clerr_);
}

void Device::ExecuteKernel(Command* cmd) {
    Kernel* kernel = cmd->kernel();
    cl_kernel clkernel = kernel->clkernel(dev_no_, clprog_);
    int dim = cmd->dim();
    size_t* ndr = cmd->ndr();
    std::map<int, KernelArg*> args = kernel->args();
    for (std::map<int, KernelArg*>::iterator it = args.begin(); it != args.end(); ++it) {
        int idx = it->first;
        KernelArg* arg = it->second;
        Mem* mem = arg->mem;
        if (mem) {
            cl_mem clmem = mem->clmem(platform_no_, clctx_);
            clerr_ = clSetKernelArg(clkernel, (cl_uint) idx, sizeof(clmem), (const void*) &clmem);
            if (clerr_ != CL_SUCCESS) _error("clerr[%d]", clerr_);
        } else {
            clerr_ = clSetKernelArg(clkernel, (cl_uint) idx, arg->size, (const void*) arg->value);
            if (clerr_ != CL_SUCCESS) _error("clerr[%d]", clerr_);
        }
    }
    _trace("kernel[%s] on %s's %s", kernel->name(), vendor_, name_);
    clerr_ = clEnqueueNDRangeKernel(clcmdq_, clkernel, (cl_uint) dim, NULL, (const size_t*) ndr, NULL, 0, NULL, NULL);
    if (clerr_ != CL_SUCCESS) _error("clerr[%d]", clerr_);
}

void Device::Wait() {
    clerr_ = clFinish(clcmdq_);
    if (clerr_ != CL_SUCCESS) _error("clerr[%d]", clerr_);
}

} /* namespace rt */
} /* namespace brisbane */
