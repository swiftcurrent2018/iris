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

    busy_ = false;
    
    timer_ = new Timer();

    clerr_ = clGetDeviceInfo(cldev_, CL_DEVICE_VENDOR, sizeof(vendor_), vendor_, NULL);
    clerr_ = clGetDeviceInfo(cldev_, CL_DEVICE_NAME, sizeof(name_), name_, NULL);
    clerr_ = clGetDeviceInfo(cldev_, CL_DEVICE_TYPE, sizeof(cltype_), &cltype_, NULL);
    clerr_ = clGetDeviceInfo(cldev_, CL_DEVICE_VERSION, sizeof(version_), version_, NULL);
    clerr_ = clGetDeviceInfo(cldev_, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compiler_available_), &compiler_available_, NULL);

    if (cltype_ == CL_DEVICE_TYPE_CPU) type_ = brisbane_device_cpu;
    else if (cltype_ == CL_DEVICE_TYPE_GPU) type_ = brisbane_device_gpu;
    else if (cltype_ == CL_DEVICE_TYPE_ACCELERATOR) {
        if (strstr(name_, "FPGA") != NULL || strstr(version_, "FPGA") != NULL) type_ = brisbane_device_fpga;
        else type_ = brisbane_device_phi;
    }
    else type_ = brisbane_device_cpu;

    _info("device[%d] vendor[%s] device[%s] type[%d] version[%s] compiler_available[%d]", dev_no_, vendor_, name_, type_, version_, compiler_available_);

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
    if (clerr_ != CL_SUCCESS) {
        cl_build_status s;
        clerr_ = clGetProgramBuildInfo(clprog_, cldev_, CL_PROGRAM_BUILD_STATUS, sizeof(s), &s, NULL);
        _clerror(clerr_);
        char log[1024];
        clerr_ = clGetProgramBuildInfo(clprog_, cldev_, CL_PROGRAM_BUILD_LOG, 1024, log, NULL);
        _clerror(clerr_);
        _error("status[%d] log:%s", s, log);
        _error("srclen[%lu] src\n%s", srclen, src);
    }
    free(src);
}

void Device::Execute(Task* task) {
    busy_ = true;
    for (int i = 0; i < task->ncmds(); i++) {
        Command* cmd = task->cmd(i);
        switch (cmd->type()) {
            case BRISBANE_CMD_KERNEL:   ExecuteKernel(cmd);     break;
            case BRISBANE_CMD_H2D:      ExecuteH2D(cmd);        break;
            case BRISBANE_CMD_D2H:      ExecuteD2H(cmd);        break;
            case BRISBANE_CMD_PRESENT:  ExecutePresent(cmd);    break;
            default: _error("cmd type[0x%x]", cmd->type());
        }
    }
    busy_ = false;
}

void Device::ExecuteKernel(Command* cmd) {
    Kernel* kernel = cmd->kernel();
    cl_kernel clkernel = kernel->clkernel(dev_no_, clprog_);
    int dim = cmd->dim();
    size_t* off = cmd->off();
    size_t* ndr = cmd->ndr();
    _trace("kernel[%s] dim[%d] off[%lu,%lu,%lu] ndr[%lu,%lu,%lu]", kernel->name(), dim, off[0], off[1], off[2], ndr[0], ndr[1], ndr[2]);
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
    _clerror(clerr_);
    double time = timer_->Stop();
    _info("kernel[%s] on dev[%d] %s time[%lf]", kernel->name(), dev_no_, name_, time);
    kernel->history()->Add(cmd, this, time);
}

void Device::ExecuteH2D(Command* cmd) {
    Mem* mem = cmd->mem();
    cl_mem clmem = mem->clmem(platform_no_, clctx_);
    size_t off = cmd->off(0);
    size_t size = cmd->size();
    void* host = cmd->host();
    _trace("devno[%d] mem[%lu] off[%lu] size[%lu] host[%p]", dev_no_, mem->uid(), off, size, host);
    mem->AddOwner(off, size, this);
    clerr_ = clEnqueueWriteBuffer(clcmdq_, clmem, CL_TRUE, off, size, host, 0, NULL, NULL);
    _clerror(clerr_);
}

void Device::ExecuteD2H(Command* cmd) {
    Mem* mem = cmd->mem();
    cl_mem clmem = mem->clmem(platform_no_, clctx_);
    size_t off = cmd->off(0);
    size_t size = cmd->size();
    void* host = cmd->host();
    _trace("devno[%d] mem[%lu] off[%lu] size[%lu] host[%p]", dev_no_, mem->uid(), off, size, host);
    clerr_ = clEnqueueReadBuffer(clcmdq_, clmem, CL_TRUE, off, size, host, 0, NULL, NULL);
    _clerror(clerr_);
}

void Device::ExecutePresent(Command* cmd) {
    Mem* mem = cmd->mem();
    cl_mem clmem = mem->clmem(platform_no_, clctx_);
    size_t off = cmd->off(0);
    size_t size = cmd->size();
    void* host = cmd->host();
    _trace("devno[%d] mem[%lu] off[%lu] size[%lu] host[%p]", dev_no_, mem->uid(), off, size, host);
    if (mem->IsOwner(off, size, this)) return;
    ExecuteH2D(cmd);
}

void Device::Wait() {
    clerr_ = clFinish(clcmdq_);
    _clerror(clerr_);
}

} /* namespace rt */
} /* namespace brisbane */
