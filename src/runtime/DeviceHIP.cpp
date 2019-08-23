#include "DeviceHIP.h"
#include "Command.h"
#include "History.h"
#include "Kernel.h"
#include "Mem.h"
#include "Reduction.h"
#include "Timer.h"
#include "Utils.h"

namespace brisbane {
namespace rt {

DeviceHIP::DeviceHIP(hipDevice_t dev, int devno, int platform) : Device(devno, platform) {
  max_arg_idx_ = 0;
  shared_mem_bytes_ = 0;
  dev_ = dev;
  strcpy(vendor_, "Advanced Micro Devices");
  err_ = hipDeviceGetName(name_, sizeof(name_), dev_);
  _hiperror(err_);
  type_ = brisbane_amd;
  err_ = hipDriverGetVersion(&driver_version_);
  _hiperror(err_);
  sprintf(version_, "AMD HIP %d", driver_version_);
  _info("device[%d] platform[%d] vendor[%s] device[%s] type[%d] version[%s] max_compute_units[%d] max_work_item_sizes[%lu,%lu,%lu]", devno_, platform_, vendor_, name_, type_, version_, max_compute_units_, max_work_item_sizes_[0], max_work_item_sizes_[1], max_work_item_sizes_[2]);
}

DeviceHIP::~DeviceHIP() {
}

int DeviceHIP::Init() {
  err_ = hipCtxCreate(&ctx_, 0, dev_);
  _hiperror(err_);

  char path[256];
  sprintf(path, "kernel.hip");
  char* src = NULL;
  size_t srclen = 0;
  if (Utils::ReadFile(path, &src, &srclen) == BRISBANE_ERR) {
    _error("dev[%d][%s] has no kernel file [%s]", devno_, name_, path);
    return BRISBANE_OK;
  }
  _trace("dev[%d][%s] kernels[%s]", devno_, name_, path);
  err_ = hipModuleLoadData(&module_, src);
  if (err_ != hipSuccess) {
    _hiperror(err_);
    _error("srclen[%lu] src\n%s", srclen, src);
    if (src) free(src);
    return BRISBANE_ERR;
  }
  if (src) free(src);
  return BRISBANE_OK;
}

int DeviceHIP::H2D(Mem* mem, size_t off, size_t size, void* host) {
  void* hipmem = mem->hipmem(devno_);
  err_ = hipMemcpyHtoD((char*) hipmem + off, host, size);
  _hiperror(err_);
  return BRISBANE_OK;
}

int DeviceHIP::D2H(Mem* mem, size_t off, size_t size, void* host) {
  void* hipmem = mem->hipmem(devno_);
  err_ = hipMemcpyDtoH(host, (char*) hipmem + off, size);
  _hiperror(err_);
  return BRISBANE_OK;
}

int DeviceHIP::KernelSetArg(Kernel* kernel, int idx, size_t size, void* value) {
  params_[idx] = value;
  if (!value) shared_mem_bytes_ += size;
  if (max_arg_idx_ < idx) max_arg_idx_ = idx;
  return BRISBANE_OK;
}

int DeviceHIP::KernelSetMem(Kernel* kernel, int idx, Mem* mem) {
  mem->hipmem(devno_);
  params_[idx] = mem->hipmems() + devno_;
  if (max_arg_idx_ < idx) max_arg_idx_ = idx;
  return BRISBANE_OK;
}

int DeviceHIP::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  hipFunction_t func = kernel->hipkernel(devno_, module_);
  int block[3] = { lws ? lws[0] : 1, lws ? lws[1] : 1, lws ? lws[2] : 1 };
  int grid[3] = { gws[0] / block[0], gws[1] / block[1], gws[2] / block[2] };
  size_t blockOff_x = off[0] / (lws ? lws[0] : 1);
  params_[max_arg_idx_ + 1] = &blockOff_x;
  _debug("grid[%d,%d,%d] block[%d,%d,%d] blockOff_x[%lu]", grid[0], grid[1], grid[2], block[0], block[1], block[2], blockOff_x);
  err_ = hipModuleLaunchKernel(func, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem_bytes_, 0, params_, NULL);
  _hiperror(err_);
  err_ = hipDeviceSynchronize();
  _hiperror(err_);
  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */

