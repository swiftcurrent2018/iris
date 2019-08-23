#ifndef BRISBANE_SRC_RT_DEVICE_CUDA_H
#define BRISBANE_SRC_RT_DEVICE_CUDA_H

#include "Device.h"

namespace brisbane {
namespace rt {

class DeviceCUDA : public Device {
public:
  DeviceCUDA(CUdevice cudev, int devno, int platform);
  ~DeviceCUDA();

  int Init();
  int H2D(Mem* mem, size_t off, size_t size, void* host);
  int D2H(Mem* mem, size_t off, size_t size, void* host);
  int KernelSetArg(Kernel* kernel, int idx, size_t size, void* value);
  int KernelSetMem(Kernel* kernel, int idx, Mem* mem);
  int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws);

private:
  CUdevice dev_;
  CUcontext ctx_;
  CUstream stream_;
  CUmodule module_;
  CUresult err_;
  unsigned int shared_mem_bytes_;
  void* params_[BRISBANE_MAX_KERNEL_NARGS];
  int max_arg_idx_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_DEVICE_CUDA_H */

