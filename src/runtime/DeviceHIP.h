#ifndef BRISBANE_SRC_RT_DEVICE_HIP_H
#define BRISBANE_SRC_RT_DEVICE_HIP_H

#include "Device.h"
#include "LoaderHIP.h"

namespace brisbane {
namespace rt {

class DeviceHIP : public Device {
public:
  DeviceHIP(LoaderHIP* ld, hipDevice_t cudev, int devno, int platform);
  ~DeviceHIP();

  int Init();
  int MemAlloc(void** mem, size_t size);
  int MemFree(void* mem);
  int MemH2D(Mem* mem, size_t off, size_t size, void* host);
  int MemD2H(Mem* mem, size_t off, size_t size, void* host);
  int KernelGet(void** kernel, const char* name);
  int KernelSetArg(Kernel* kernel, int idx, size_t size, void* value);
  int KernelSetMem(Kernel* kernel, int idx, Mem* mem, size_t off);
  int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws);
  int Synchronize();
  int AddCallback(Task* task);

private:
  LoaderHIP* ld_;
  hipDevice_t dev_;
  hipCtx_t ctx_;
  hipModule_t module_;
  hipError_t err_;
  unsigned int shared_mem_bytes_;
  void* params_[BRISBANE_MAX_KERNEL_NARGS];
  int max_arg_idx_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_DEVICE_HIP_H */

