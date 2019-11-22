#include "DeviceHIP.h"
#include "Debug.h"
#include "Command.h"
#include "History.h"
#include "Kernel.h"
#include "LoaderHIP.h"
#include "Mem.h"
#include "Reduction.h"
#include "Timer.h"
#include "Utils.h"

namespace brisbane {
namespace rt {

DeviceHIP::DeviceHIP(LoaderHIP* ld, hipDevice_t dev, int devno, int platform) : Device(devno, platform) {
  ld_ = ld;
  max_arg_idx_ = 0;
  shared_mem_bytes_ = 0;
  dev_ = dev;
  strcpy(vendor_, "Advanced Micro Devices");
  err_ = ld_->hipDeviceGetName(name_, sizeof(name_), dev_);
  _hiperror(err_);
  type_ = brisbane_amd;
  err_ = ld_->hipDriverGetVersion(&driver_version_);
  _hiperror(err_);
  sprintf(version_, "AMD HIP %d", driver_version_);
  _info("device[%d] platform[%d] vendor[%s] device[%s] type[%d] version[%s] max_compute_units[%d] max_work_item_sizes[%lu,%lu,%lu]", devno_, platform_, vendor_, name_, type_, version_, max_compute_units_, max_work_item_sizes_[0], max_work_item_sizes_[1], max_work_item_sizes_[2]);
}

DeviceHIP::~DeviceHIP() {
}

int DeviceHIP::Init() {
  err_ = ld_->hipCtxCreate(&ctx_, 0, dev_);
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
  err_ = ld_->hipModuleLoad(&module_, "kernel.hip");
  if (err_ != hipSuccess) {
    _hiperror(err_);
    _error("srclen[%lu] src\n%s", srclen, src);
    if (src) free(src);
    return BRISBANE_ERR;
  }
  if (src) free(src);
  return BRISBANE_OK;
}

int DeviceHIP::MemAlloc(void** mem, size_t size) {
  void** hipmem = mem;
  err_ = ld_->hipMalloc(hipmem, size);
  _hiperror(err_);
  return BRISBANE_OK;
}

int DeviceHIP::MemFree(void* mem) {
  void* hipmem = mem;
  err_ = ld_->hipFree(hipmem);
  _hiperror(err_);
  return BRISBANE_OK;
}

int DeviceHIP::MemH2D(Mem* mem, size_t off, size_t size, void* host) {
  void* hipmem = mem->arch(this);
  err_ = ld_->hipMemcpyHtoD((char*) hipmem + off, host, size);
  _hiperror(err_);
  return BRISBANE_OK;
}

int DeviceHIP::MemD2H(Mem* mem, size_t off, size_t size, void* host) {
  void* hipmem = mem->arch(this);
  err_ = ld_->hipMemcpyDtoH(host, (char*) hipmem + off, size);
  _hiperror(err_);
  return BRISBANE_OK;
}

int DeviceHIP::KernelGet(void** kernel, const char* name) {
  hipFunction_t* hipkernel = (hipFunction_t*) kernel;
  err_ = ld_->hipModuleGetFunction(hipkernel, module_, name);
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
  mem->arch(this);
  params_[idx] = mem->archs() + devno_;
  if (max_arg_idx_ < idx) max_arg_idx_ = idx;
  return BRISBANE_OK;
}

int DeviceHIP::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  hipFunction_t func = (hipFunction_t) kernel->arch(this);
  int block[3] = { lws ? (int) lws[0] : 1, lws ? (int) lws[1] : 1, lws ? (int) lws[2] : 1 };
  int grid[3] = { (int) (gws[0] / block[0]), (int) (gws[1] / block[1]), (int) (gws[2] / block[2]) };
  size_t blockOff_x = off[0] / (lws ? lws[0] : 1);
  params_[max_arg_idx_ + 1] = &blockOff_x;
  _trace("grid[%d,%d,%d] block[%d,%d,%d] blockOff_x[%lu]", grid[0], grid[1], grid[2], block[0], block[1], block[2], blockOff_x);
  err_ = ld_->hipModuleLaunchKernel(func, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem_bytes_, 0, params_, NULL);
  _hiperror(err_);
  err_ = ld_->hipDeviceSynchronize();
  _hiperror(err_);
  for (int i = 0; i < BRISBANE_MAX_KERNEL_NARGS; i++) params_[i] = NULL;
  max_arg_idx_ = 0;
  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */

