#include "Device.h"
#include "History.h"
#include "Kernel.h"
#include "Mem.h"
#include "Timer.h"
#include "Utils.h"

namespace brisbane {
namespace rt {

Device::Device(cl_device_id cldev, cl_context clctx, int dev_no, int platform_no) {
    cldev_ = cldev;
    clctx_ = clctx;
    dev_no_ = dev_no;
    platform_no_ = platform_no;
    
    timer_ = new Timer();

    clerr_ = clGetDeviceInfo(cldev_, CL_DEVICE_VENDOR, sizeof(vendor_), vendor_, NULL);
    clerr_ = clGetDeviceInfo(cldev_, CL_DEVICE_NAME, sizeof(name_), name_, NULL);
    clerr_ = clGetDeviceInfo(cldev_, CL_DEVICE_TYPE, sizeof(cltype_), &cltype_, NULL);
    clerr_ = clGetDeviceInfo(cldev_, CL_DEVICE_VERSION, sizeof(version_), version_, NULL);

    if (cltype_ == CL_DEVICE_TYPE_CPU) type_ = brisbane_device_cpu;
    else if (cltype_ == CL_DEVICE_TYPE_GPU) type_ = brisbane_device_gpu;
    else if (cltype_ == CL_DEVICE_TYPE_ACCELERATOR) {
        if (strstr(name_, "FPGA") != NULL || strstr(version_, "FPGA") != NULL) type_ = brisbane_device_fpga;
        else type_ = brisbane_device_phi;
    }
    else type_ = brisbane_device_cpu;

    _info("device[%d] vendor[%s] device[%s] type[%d] version[%s]", dev_no_, vendor_, name_, type_, version_);

    clcmdq_ = clCreateCommandQueue(clctx_, cldev_, 0, &clerr_);
    _clerror(clerr_);

    BuildProgram();
}

Device::~Device() {
    delete timer_;
}

void Device::BuildProgram() {
    cl_int status;
    char path[256];
    memset(path, 0, 256);
    sprintf(path, "kernel-%s",
            type_ == brisbane_device_cpu  ? "cpu.cl"  :
            type_ == brisbane_device_gpu  ? "gpu.cl"  :
            type_ == brisbane_device_phi  ? "phi.cl"  :
            type_ == brisbane_device_fpga ? "fpga.aocx" : "default.cl");
    char* src = NULL;
    size_t srclen = 0;
    Utils::ReadFile(path, &src, &srclen);
    if (srclen == 0) return;
    if (type_ == brisbane_device_fpga) clprog_ = clCreateProgramWithBinary(clctx_, 1, &cldev_, (const size_t*) &srclen, (const unsigned char**) &src, &status, &clerr_);
    else clprog_ = clCreateProgramWithSource(clctx_, 1, (const char**) &src, (const size_t*) &srclen, &clerr_);
    _clerror(clerr_);
    clerr_ = clBuildProgram(clprog_, 1, &cldev_, "", NULL, NULL);
    _clerror(clerr_);
    free(src);
}

void Device::Execute(Task* task) {
    for (int i = 0; i < task->ncmds(); i++) {
        Command* cmd = task->cmd(i);
        switch (cmd->type()) {
            case BRISBANE_CMD_KERNEL:   ExecuteKernel(cmd);     break;
            case BRISBANE_CMD_H2D:      ExecuteH2D(cmd);        break;
            case BRISBANE_CMD_D2H:      ExecuteD2H(cmd);        break;
            case BRISBANE_CMD_PRESENT:                          break;
            default: _error("cmd type[0x%x]", cmd->type());
        }
    }
}

void Device::ExecuteKernel(Command* cmd) {
    Kernel* kernel = cmd->kernel();
    cl_kernel clkernel = kernel->clkernel(dev_no_, clprog_);
    int dim = cmd->dim();
    size_t* off = cmd->off();
    size_t* ndr = cmd->ndr();
    std::map<int, KernelArg*>* args = kernel->args();
    for (std::map<int, KernelArg*>::iterator it = args->begin(); it != args->end(); ++it) {
        int idx = it->first;
        KernelArg* arg = it->second;
        Mem* mem = arg->mem;
        if (mem) {
            if (arg->mode & brisbane_wr) mem->SetOwner(this);
            cl_mem clmem = mem->clmem(platform_no_, clctx_);
            clerr_ = clSetKernelArg(clkernel, (cl_uint) idx, sizeof(clmem), (const void*) &clmem);
            _clerror(clerr_);
        } else {
            clerr_ = clSetKernelArg(clkernel, (cl_uint) idx, arg->size, (const void*) arg->value);
            _clerror(clerr_);
        }
    }
    timer_->Start();
    if (type_ == brisbane_device_fpga) {
        if (off[0] != 0 || off[1] != 0 || off[2] != 0)
            _todo("%s", "global_work_offset shoule be set to not NULL. Upgrade Intel FPGA SDK for OpenCL Pro Edition Version 19.1");
        clerr_ = clEnqueueNDRangeKernel(clcmdq_, clkernel, (cl_uint) dim, NULL, (const size_t*) ndr, NULL, 0, NULL, NULL);
    } else {
        clerr_ = clEnqueueNDRangeKernel(clcmdq_, clkernel, (cl_uint) dim, (const size_t*) off, (const size_t*) ndr, NULL, 0, NULL, NULL);
    }
    _clerror(clerr_);
    clerr_ = clFinish(clcmdq_);
    double time = timer_->Stop();
    _trace("kernel[%s] on %s time[%lf]", kernel->name(), name_, time);
    kernel->history()->Add(cmd, this, time);
    _clerror(clerr_);
}

void Device::ExecuteH2D(Command* cmd) {
    Mem* mem = cmd->mem();
    cl_mem clmem = mem->clmem(platform_no_, clctx_);
    size_t off = cmd->off(0);
    size_t size = cmd->size();
    void* host = cmd->host();
    mem->AddOwner(this);
    clerr_ = clEnqueueWriteBuffer(clcmdq_, clmem, CL_TRUE, off, size, host, 0, NULL, NULL);
    _clerror(clerr_);
}

void Device::ExecuteD2H(Command* cmd) {
    Mem* mem = cmd->mem();
    cl_mem clmem = mem->clmem(platform_no_, clctx_);
    size_t off = cmd->off(0);
    size_t size = cmd->size();
    void* host = cmd->host();

    clerr_ = clEnqueueReadBuffer(clcmdq_, clmem, CL_TRUE, off, size, host, 0, NULL, NULL);
    _clerror(clerr_);
}

void Device::Wait() {
    clerr_ = clFinish(clcmdq_);
    _clerror(clerr_);
}

} /* namespace rt */
} /* namespace brisbane */
