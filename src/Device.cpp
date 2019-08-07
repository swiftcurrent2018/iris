#include "Device.h"
#include "History.h"
#include "Kernel.h"
#include "Mem.h"
#include "Reduction.h"
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
  clerr_ = clGetDeviceInfo(cldev_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units_), &max_compute_units_, NULL);
  clerr_ = clGetDeviceInfo(cldev_, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes_), max_work_item_sizes_, NULL);
  clerr_ = clGetDeviceInfo(cldev_, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compiler_available_), &compiler_available_, NULL);

  if (cltype_ == CL_DEVICE_TYPE_CPU) type_ = brisbane_device_cpu;
  else if (cltype_ == CL_DEVICE_TYPE_GPU) {
    type_ = brisbane_device_gpu;
    if (strcasestr(vendor_, "NVIDIA")) type_ = brisbane_device_nvidia;
    else if (strcasestr(vendor_, "AMD")) type_ = brisbane_device_amd;
  }
  else if (cltype_ == CL_DEVICE_TYPE_ACCELERATOR) {
    if (strstr(name_, "FPGA") != NULL || strstr(version_, "FPGA") != NULL) type_ = brisbane_device_fpga;
    else type_ = brisbane_device_phi;
  }
  else type_ = brisbane_device_cpu;

  _info("device[%d] vendor[%s] device[%s] type[%d] version[%s] max_compute_units[%d] max_work_item_sizes[%lu,%lu,%lu] compiler_available[%d]", dev_no_, vendor_, name_, type_, version_, max_compute_units_, max_work_item_sizes_[0], max_work_item_sizes_[1], max_work_item_sizes_[2], compiler_available_);

  clcmdq_ = clCreateCommandQueue(clctx_, cldev_, 0, &clerr_);
  _clerror(clerr_);

  enabled_ = BuildProgram();
}

Device::~Device() {
  delete timer_;
}

bool Device::BuildProgram() {
  cl_int status;
  char path[256];
  memset(path, 0, 256);
  sprintf(path, "kernel-%s",
    type_ == brisbane_device_cpu    ? "cpu.cl"  :
    type_ == brisbane_device_nvidia ? "nvidia.cl"  :
    type_ == brisbane_device_amd    ? "amd.cl"  :
    type_ == brisbane_device_gpu    ? "gpu.cl"  :
    type_ == brisbane_device_phi    ? "phi.cl"  :
    type_ == brisbane_device_fpga   ? "fpga.aocx" : "default.cl");
  char* src = NULL;
  size_t srclen = 0;
  if (Utils::ReadFile(path, &src, &srclen) == BRISBANE_ERR) {
    memset(path, 0, 256);
    sprintf(path, "kernel.cl");
    Utils::ReadFile(path, &src, &srclen);
  }
  if (srclen == 0) {
    _error("dev[%d][%s] has no kernel file", dev_no_, name_);
    return false;
  }
  _trace("dev[%d][%s] kernels[%s]", dev_no_, name_, path);
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
    size_t log_size;
    clerr_ = clGetProgramBuildInfo(clprog_, cldev_, CL_PROGRAM_BUILD_LOG, 1024, log, &log_size);
    _clerror(clerr_);
    _error("status[%d] log_size[%lu] log:%s", s, log_size, log);
    _error("srclen[%lu] src\n%s", srclen, src);
    free(src);
    return false;
  }
  free(src);
  return true;
}

void Device::Execute(Task* task) {
  busy_ = true;
  for (int i = 0; i < task->ncmds(); i++) {
    Command* cmd = task->cmd(i);
    switch (cmd->type()) {
      case BRISBANE_CMD_KERNEL:       ExecuteKernel(cmd);     break;
      case BRISBANE_CMD_H2D:          ExecuteH2D(cmd);        break;
      case BRISBANE_CMD_D2H:          ExecuteD2H(cmd);        break;
      case BRISBANE_CMD_PRESENT:      ExecutePresent(cmd);    break;
      case BRISBANE_CMD_RELEASE_MEM:  ExecuteReleaseMem(cmd); break;
      default: _error("cmd type[0x%x]", cmd->type());
    }
  }
  _info("task[%lu] complete", task->uid());
  busy_ = false;
}

void Device::ExecuteKernel(Command* cmd) {
  Kernel* kernel = cmd->kernel();
  cl_kernel clkernel = kernel->clkernel(dev_no_, clprog_);
  int dim = cmd->dim();
  size_t* off = cmd->off();
  size_t* gws = cmd->ndr();
  size_t gws0 = gws[0];
  size_t* lws = NULL;
  bool reduction = false;
  int max_idx = 0;
  std::map<int, KernelArg*>* args = cmd->kernel_args();
  for (std::map<int, KernelArg*>::iterator I = args->begin(), E = args->end(); I != E; ++I) {
    int idx = I->first;
    if (idx > max_idx) max_idx = idx;
    KernelArg* arg = I->second;
    Mem* mem = arg->mem;
    if (mem) {
      if (arg->mode & brisbane_wr) mem->SetOwner(this);
      if (mem->mode() & brisbane_reduction) {
        lws = (size_t*) alloca(3 * sizeof(size_t));
        lws[0] = 1;
        lws[1] = 1;
        lws[2] = 1;
        while (max_compute_units_ * lws[0] < gws[0]) lws[0] <<= 1;
        while (max_work_item_sizes_[0] / 4 < lws[0]) lws[0] >>= 1;
        size_t expansion = (gws[0] + lws[0] - 1) / lws[0];
        gws[0] = lws[0] * expansion;
        mem->Expand(expansion);
        clerr_ = clSetKernelArg(clkernel, (cl_uint) idx + 1, lws[0] * mem->type_size(), NULL);
        _clerror(clerr_);
        reduction = true;
        if (idx + 1 > max_idx) max_idx = idx + 1;
      }
      cl_mem clmem = mem->clmem(platform_no_, clctx_);
      clerr_ = clSetKernelArg(clkernel, (cl_uint) idx, sizeof(clmem), (const void*) &clmem);
      _clerror(clerr_);
    } else {
      clerr_ = clSetKernelArg(clkernel, (cl_uint) idx, arg->size, (const void*) arg->value);
      _clerror(clerr_);
    }
  }
  if (reduction) {
    _trace("max_idx+1[%d] gws[%lu]", max_idx + 1, gws0);
    clerr_ = clSetKernelArg(clkernel, (cl_uint) max_idx + 1, sizeof(size_t), &gws0);
    _clerror(clerr_);
  }
  _trace("devno[%d][%s] kernel[%s] dim[%d] off[%lu,%lu,%lu] gws[%lu,%lu,%lu] lws[%lu,%lu,%lu]", dev_no_, name_, kernel->name(), dim, off[0], off[1], off[2], gws[0], gws[1], gws[2], lws ? lws[0] : 0, lws ? lws[1] : 0, lws ? lws[2] : 0);
  if (lws && (lws[0] > gws[0] || lws[1] > gws[1] || lws[2] > gws[2])) _error("gws[%lu,%lu,%lu] and lws[%lu,%lu,%lu]", gws[0], gws[1], gws[2], lws[0], lws[1], lws[2]);
  timer_->Start(11);
  if (type_ == brisbane_device_fpga) {
    if (off[0] != 0 || off[1] != 0 || off[2] != 0)
      _todo("%s", "global_work_offset shoule be set to not NULL. Upgrade Intel FPGA SDK for OpenCL Pro Edition Version 19.1");
    clerr_ = clEnqueueNDRangeKernel(clcmdq_, clkernel, (cl_uint) dim, NULL, (const size_t*) gws, (const size_t*) lws, 0, NULL, NULL);
  } else {
    clerr_ = clEnqueueNDRangeKernel(clcmdq_, clkernel, (cl_uint) dim, (const size_t*) off, (const size_t*) gws, (const size_t*) lws, 0, NULL, NULL);
  }
  _clerror(clerr_);
  clerr_ = clFinish(clcmdq_);
  _clerror(clerr_);
  double time = timer_->Stop(11);
  _trace("devno[%d][%s] kernel[%s] time[%lf]", dev_no_, name_, kernel->name(), time);
  kernel->history()->AddKernel(cmd, this, time);
}

void Device::ExecuteH2D(Command* cmd) {
  Mem* mem = cmd->mem();
  cl_mem clmem = mem->clmem(platform_no_, clctx_);
  size_t off = cmd->off(0);
  size_t size = cmd->size();
  void* host = cmd->host();
  _trace("devno[%d][%s] mem[%lu] clmcm[%p] off[%lu] size[%lu] host[%p]", dev_no_, name_, mem->uid(), clmem, off, size, host);
  mem->AddOwner(off, size, this);
  timer_->Start(12);
  clerr_ = clEnqueueWriteBuffer(clcmdq_, clmem, CL_TRUE, off, size, host, 0, NULL, NULL);
  _clerror(clerr_);
  double time = timer_->Stop(12);
  Command* cmd_kernel = cmd->task()->cmd_kernel();
  if (cmd_kernel) cmd_kernel->kernel()->history()->AddH2D(cmd, this, time);
  else Platform::GetPlatform()->null_kernel()->history()->AddH2D(cmd, this, time);
}

void Device::ExecuteD2H(Command* cmd) {
  Mem* mem = cmd->mem();
  int mode = mem->mode();
  cl_mem clmem = mem->clmem(platform_no_, clctx_);
  size_t off = cmd->off(0);
  size_t size = cmd->size();
  int expansion = mem->expansion();
  void* host = cmd->host();
  _trace("devno[%d][%s] mem[%lu] off[%lu] size[%lu] expansion[%d] host[%p]", dev_no_, name_, mem->uid(), off, size, expansion, host);
  timer_->Start(13);
  if (mode & brisbane_reduction) {
    clerr_ = clEnqueueReadBuffer(clcmdq_, clmem, CL_TRUE, off, mem->size() * expansion, mem->host_inter(), 0, NULL, NULL);
    Reduction::GetInstance()->Reduce(mem, host, size);
  } else clerr_ = clEnqueueReadBuffer(clcmdq_, clmem, CL_TRUE, off, size, host, 0, NULL, NULL);
  _clerror(clerr_);
  double time = timer_->Stop(13);
  Command* cmd_kernel = cmd->task()->cmd_kernel();
  if (cmd_kernel) cmd_kernel->kernel()->history()->AddD2H(cmd, this, time);
  else Platform::GetPlatform()->null_kernel()->history()->AddD2H(cmd, this, time);
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

void Device::ExecuteReleaseMem(Command* cmd) {
  Mem* mem = cmd->mem();
  delete mem; 
}

void Device::Wait() {
  clerr_ = clFinish(clcmdq_);
  _clerror(clerr_);
}

} /* namespace rt */
} /* namespace brisbane */
