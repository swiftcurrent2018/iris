#ifndef BRISBANE_SRC_RT_DEVICE_HIP_H
#define BRISBANE_SRC_RT_DEVICE_HIP_H

#include "Device.h"

namespace brisbane {
namespace rt {

class DeviceHIP : public Device {
public:
  DeviceHIP(hipDevice_t cudev, int devno, int platform);
  ~DeviceHIP();

  int Init();
  int H2D(Mem* mem, size_t off, size_t size, void* host);
  int D2H(Mem* mem, size_t off, size_t size, void* host);
  int KernelSetArg(Kernel* kernel, int idx, size_t size, void* value);
  int KernelSetMem(Kernel* kernel, int idx, Mem* mem);
  int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws);

private:
  hipDevice_t dev_;
  hipCtx_t ctx_;
  hipModule_t module_;
  hipError_t err_;
  unsigned int shared_mem_bytes_;
  void* params_[BRISBANE_MAX_KERNEL_NARGS];
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_DEVICE_HIP_H */

