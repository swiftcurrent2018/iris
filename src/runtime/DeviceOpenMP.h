#ifndef BRISBANE_SRC_RT_DEVICE_OPENMP_H
#define BRISBANE_SRC_RT_DEVICE_OPENMP_H

#include "Device.h"

namespace brisbane {
namespace rt {

class DeviceOpenMP : public Device {
public:
  DeviceOpenMP(int devno, int platform);
  ~DeviceOpenMP();

  int Init();
  int H2D(Mem* mem, size_t off, size_t size, void* host);
  int D2H(Mem* mem, size_t off, size_t size, void* host);
  int KernelLaunchInit(Kernel* kernel);
  int KernelSetArg(Kernel* kernel, int idx, size_t size, void* value);
  int KernelSetMem(Kernel* kernel, int idx, Mem* mem);
  int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws);

private:
  void* handle_;
  int dlerr_;

  int (*kernel_)(const char* name);
  int (*setarg_)(int idx, size_t size, void* value);
  int (*setmem_)(int idx, void* mem);
  int (*launch_)(int dim, size_t off, size_t ndr);

  int GetProcessorNameIntel(char* cpuinfo);
  int GetProcessorNameARM(char* cpuinfo);
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_DEVICE_OPENMP_H */

