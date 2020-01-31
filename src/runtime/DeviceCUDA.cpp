#include "DeviceCUDA.h"
#include "Debug.h"
#include "Command.h"
#include "History.h"
#include "Kernel.h"
#include "LoaderCUDA.h"
#include "Mem.h"
#include "Reduction.h"
#include "Task.h"
#include "Timer.h"
#include "Utils.h"

namespace brisbane {
namespace rt {

DeviceCUDA::DeviceCUDA(LoaderCUDA* ld, CUdevice cudev, int devno, int platform) : Device(devno, platform) {
  ld_ = ld;
  max_arg_idx_ = 0;
  shared_mem_bytes_ = 0;
  dev_ = cudev;
  strcpy(vendor_, "NVIDIA Corporation");
  err_ = ld_->cuDeviceGetName(name_, sizeof(name_), dev_);
  _cuerror(err_);
  type_ = brisbane_nvidia;
  err_ = ld_->cuDriverGetVersion(&driver_version_);
  _cuerror(err_);
  sprintf(version_, "NVIDIA CUDA %d", driver_version_);
  int tb, mc, bx, by, bz, dx, dy, dz, ck;
  err_ = ld_->cuDeviceGetAttribute(&tb, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev_);
  err_ = ld_->cuDeviceGetAttribute(&mc, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev_);
  err_ = ld_->cuDeviceGetAttribute(&bx, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, dev_);
  err_ = ld_->cuDeviceGetAttribute(&by, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, dev_);
  err_ = ld_->cuDeviceGetAttribute(&bz, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, dev_);
  err_ = ld_->cuDeviceGetAttribute(&dx, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev_);
  err_ = ld_->cuDeviceGetAttribute(&dy, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, dev_);
  err_ = ld_->cuDeviceGetAttribute(&dz, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, dev_);
  err_ = ld_->cuDeviceGetAttribute(&ck, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, dev_);
  max_work_group_size_ = tb;
  max_compute_units_ = mc;
  max_work_item_sizes_[0] = (size_t) bx * (size_t) dx;
  max_work_item_sizes_[1] = (size_t) by * (size_t) dy;
  max_work_item_sizes_[2] = (size_t) bz * (size_t) dz;
  max_block_dims_[0] = bx;
  max_block_dims_[1] = by;
  max_block_dims_[2] = bz;
  _info("device[%d] platform[%d] vendor[%s] device[%s] type[%d] version[%s] max_compute_units[%d] max_work_group_size_[%lu] max_work_item_sizes[%lu,%lu,%lu] max_block_dims[%d,%d,%d] concurrent_kernels[%d]", devno_, platform_, vendor_, name_, type_, version_, max_compute_units_, max_work_group_size_, max_work_item_sizes_[0], max_work_item_sizes_[1], max_work_item_sizes_[2], max_block_dims_[0], max_block_dims_[1], max_block_dims_[2], ck);
}

DeviceCUDA::~DeviceCUDA() {
}


int DeviceCUDA::Init() {
  err_ = ld_->cuCtxCreate(&ctx_, CU_CTX_SCHED_AUTO, dev_);
  _cuerror(err_);
  for (int i = 0; i < nqueues_; i++) {
    err_ = ld_->cuStreamCreate(streams_ + i, CU_STREAM_DEFAULT);
    _cuerror(err_);
  }

  char path[256];
  sprintf(path, "kernel.ptx");
  char* src = NULL;
  size_t srclen = 0;
  if (Utils::ReadFile(path, &src, &srclen) == BRISBANE_ERR) {
    _error("dev[%d][%s] has no kernel file [%s]", devno_, name_, path);
    return BRISBANE_OK;
  }
  _trace("dev[%d][%s] kernels[%s]", devno_, name_, path);
  err_ = ld_->cuModuleLoad(&module_, "kernel.ptx");
  if (err_ != CUDA_SUCCESS) {
    _cuerror(err_);
    _error("srclen[%lu] src\n%s", srclen, src);
    if (src) free(src);
    return BRISBANE_ERR;
  }
  if (src) free(src);
  return BRISBANE_OK;
}

int DeviceCUDA::MemAlloc(void** mem, size_t size) {
  CUdeviceptr* cumem = (CUdeviceptr*) mem;
  err_ = ld_->cuMemAlloc(cumem, size);
  _cuerror(err_);
  return BRISBANE_OK;
}

int DeviceCUDA::MemFree(void* mem) {
  CUdeviceptr cumem = (CUdeviceptr) mem;
  err_ = ld_->cuMemFree(cumem);
  _cuerror(err_);
  return BRISBANE_OK;
}

int DeviceCUDA::MemH2D(Mem* mem, size_t off, size_t size, void* host) {
  CUdeviceptr cumem = (CUdeviceptr) mem->arch(this);
  _trace("mem[%lu] off[%lu] size[%lu] host[%p] q[%d]", mem->uid(), off, size, host, q_);
  err_ = ld_->cuMemcpyHtoDAsync(cumem + off, host, size, streams_[q_]);
  _cuerror(err_);
  return BRISBANE_OK;
}

int DeviceCUDA::MemD2H(Mem* mem, size_t off, size_t size, void* host) {
  CUdeviceptr cumem = (CUdeviceptr) mem->arch(this);
  _trace("mem[%lu] off[%lu] size[%lu] host[%p] q[%d]", mem->uid(), off, size, host, q_);
  err_ = ld_->cuMemcpyDtoHAsync(host, cumem + off, size, streams_[q_]);
  _cuerror(err_);
  return BRISBANE_OK;
}

int DeviceCUDA::KernelGet(void** kernel, const char* name) {
  CUfunction* cukernel = (CUfunction*) kernel;
  err_ = ld_->cuModuleGetFunction(cukernel, module_, name);
  _cuerror(err_);
  return BRISBANE_OK;
}

int DeviceCUDA::KernelSetArg(Kernel* kernel, int idx, size_t size, void* value) {
  params_[idx] = value;
  if (!value) shared_mem_bytes_ += size;
  if (max_arg_idx_ < idx) max_arg_idx_ = idx;
  return BRISBANE_OK;
}

int DeviceCUDA::KernelSetMem(Kernel* kernel, int idx, Mem* mem, size_t off) {
  mem->arch(this);
  if (off) {
    *(mem->archs_off() + devno_) = (void*) ((CUdeviceptr) mem->arch(this) + off);
    params_[idx] = mem->archs_off() + devno_;
  } else params_[idx] = mem->archs() + devno_;
  if (max_arg_idx_ < idx) max_arg_idx_ = idx;
  return BRISBANE_OK;
}

int DeviceCUDA::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  CUfunction cukernel = (CUfunction) kernel->arch(this);
  int block[3] = { lws ? (int) lws[0] : 1, lws ? (int) lws[1] : 1, lws ? (int) lws[2] : 1 };
  if (!lws) {
    while (max_compute_units_ * block[0] < gws[0]) block[0] <<= 1;
    while (block[0] > max_block_dims_[0]) block[0] >>= 1;
  }
  int grid[3] = { (int) (gws[0] / block[0]), (int) (gws[1] / block[1]), (int) (gws[2] / block[2]) };
  size_t blockOff_x = off[0] / block[0];
  params_[max_arg_idx_ + 1] = &blockOff_x;
  _trace("kernel[%s] dim[%d] grid[%d,%d,%d] block[%d,%d,%d] blockOff_x[%lu] q[%d]", kernel->name(), dim, grid[0], grid[1], grid[2], block[0], block[1], block[2], blockOff_x, q_);
  err_ = ld_->cuLaunchKernel(cukernel, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem_bytes_, streams_[q_], params_, NULL);
  _cuerror(err_);
  /*
  err_ = ld_->cuStreamSynchronize(streams_[q_]);
  _cuerror(err_);
  */
  for (int i = 0; i < BRISBANE_MAX_KERNEL_NARGS; i++) params_[i] = NULL;
  max_arg_idx_ = 0;
  return BRISBANE_OK;
}

int DeviceCUDA::Synchronize() {
  err_ = ld_->cuCtxSynchronize();
  _cuerror(err_);
  return BRISBANE_OK;
}

int DeviceCUDA::AddCallback(Task* task) {
  _debug("task[%lu]", task->uid());
  err_ = ld_->cuStreamAddCallback(streams_[q_], DeviceCUDA::Callback, task, 0);
  _cuerror(err_);
  return BRISBANE_OK;
}

void DeviceCUDA::Callback(CUstream stream, CUresult status, void* data) {
  Task* task = (Task*) data;
  _debug("task[%lu]", task->uid());
  task->Complete();
}

} /* namespace rt */
} /* namespace brisbane */

