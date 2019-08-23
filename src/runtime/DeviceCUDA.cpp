#include "DeviceCUDA.h"
#include "Command.h"
#include "History.h"
#include "Kernel.h"
#include "Mem.h"
#include "Reduction.h"
#include "Timer.h"
#include "Utils.h"

namespace brisbane {
namespace rt {

DeviceCUDA::DeviceCUDA(CUdevice cudev, int devno, int platform) : Device(devno, platform) {
  shared_mem_bytes_ = 0;
  dev_ = cudev;
  strcpy(vendor_, "NVIDIA Corporation");
  err_ = cuDeviceGetName(name_, sizeof(name_), dev_);
  _cuerror(err_);
  type_ = brisbane_nvidia;
  err_ = cuDriverGetVersion(&driver_version_);
  _cuerror(err_);
  sprintf(version_, "NVIDIA CUDA %d", driver_version_);
  int tb, bx, by, bz, dx, dy, dz;
  err_ = cuDeviceGetAttribute(&tb, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev_);
  err_ = cuDeviceGetAttribute(&bx, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, dev_);
  err_ = cuDeviceGetAttribute(&by, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, dev_);
  err_ = cuDeviceGetAttribute(&bz, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, dev_);
  err_ = cuDeviceGetAttribute(&dx, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev_);
  err_ = cuDeviceGetAttribute(&dy, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, dev_);
  err_ = cuDeviceGetAttribute(&dz, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, dev_);
  max_compute_units_ = tb;
  max_work_item_sizes_[0] = (size_t) bx * (size_t) dx;
  max_work_item_sizes_[1] = (size_t) by * (size_t) dy;
  max_work_item_sizes_[2] = (size_t) bz * (size_t) dz;
  _info("device[%d] platform[%d] vendor[%s] device[%s] type[%d] version[%s] max_compute_units[%d] max_work_item_sizes[%lu,%lu,%lu]", devno_, platform_, vendor_, name_, type_, version_, max_compute_units_, max_work_item_sizes_[0], max_work_item_sizes_[1], max_work_item_sizes_[2]);
}

DeviceCUDA::~DeviceCUDA() {
}

int DeviceCUDA::Init() {
  err_ = cuCtxCreate(&ctx_, CU_CTX_SCHED_AUTO, dev_);
  _cuerror(err_);

  err_ = cuStreamCreate(&stream_, CU_STREAM_DEFAULT);
  _cuerror(err_);

  char path[256];
  sprintf(path, "kernel.ptx");
  char* src = NULL;
  size_t srclen = 0;
  if (Utils::ReadFile(path, &src, &srclen) == BRISBANE_ERR) {
    _error("dev[%d][%s] has no kernel file [%s]", devno_, name_, path);
    return BRISBANE_OK;
  }
  _trace("dev[%d][%s] kernels[%s]", devno_, name_, path);
  err_ = cuModuleLoadData(&module_, src);
  if (err_ != CUDA_SUCCESS) {
    _cuerror(err_);
    _error("srclen[%lu] src\n%s", srclen, src);
    if (src) free(src);
    return BRISBANE_ERR;
  }
  if (src) free(src);
  return BRISBANE_OK;
}

int DeviceCUDA::H2D(Mem* mem, size_t off, size_t size, void* host) {
  CUdeviceptr cumem = mem->cumem(devno_);
  err_ = cuMemcpyHtoD(cumem + off, host, size);
  _cuerror(err_);
  return BRISBANE_OK;
}

int DeviceCUDA::D2H(Mem* mem, size_t off, size_t size, void* host) {
  CUdeviceptr cumem = mem->cumem(devno_);
  err_ = cuMemcpyDtoH(host, cumem + off, size);
  _cuerror(err_);
  return BRISBANE_OK;
}

int DeviceCUDA::KernelSetArg(Kernel* kernel, int idx, size_t size, void* value) {
  params_[idx] = value;
  if (!value) shared_mem_bytes_ += size;
  return BRISBANE_OK;
}

int DeviceCUDA::KernelSetMem(Kernel* kernel, int idx, Mem* mem) {
  mem->cumem(devno_);
  params_[idx] = mem->cumems() + devno_;
  return BRISBANE_OK;
}

int DeviceCUDA::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  CUfunction func = kernel->cukernel(devno_, module_);
  int block[3] = { lws ? lws[0] : 1, lws ? lws[1] : 1, lws ? lws[2] : 1 };
  int grid[3] = { gws[0] / block[0], gws[1] / block[1], gws[2] / block[2] };
  _debug("grid[%d,%d,%d] block[%d,%d,%d]", grid[0], grid[1], grid[2], block[0], block[1], block[2]);
  err_ = cuLaunchKernel(func, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem_bytes_, stream_, params_, NULL);
  _cuerror(err_);
  err_ = cuStreamSynchronize(stream_);
  _cuerror(err_);
  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */

