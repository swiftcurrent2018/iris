#ifndef BRISBANE_SRC_RT_DEVICE_OPENCL_H
#define BRISBANE_SRC_RT_DEVICE_OPENCL_H

#include "Device.h"

namespace brisbane {
namespace rt {

class DeviceOpenCL : public Device {
public:
  DeviceOpenCL(cl_device_id cldev, cl_context clctx, int devno, int platform);
  ~DeviceOpenCL();

  int Init();
  int H2D(Mem* mem, size_t off, size_t size, void* host);
  int D2H(Mem* mem, size_t off, size_t size, void* host);
  int KernelSetArg(Kernel* kernel, int idx, size_t size, void* value);
  int KernelSetMem(Kernel* kernel, int idx, Mem* mem);
  int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws);

private:
  cl_device_id cldev_;
  cl_context clctx_;
  cl_command_queue clcmdq_;
  cl_program clprog_;
  cl_device_type cltype_;
  cl_bool compiler_available_;
  cl_int err_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_DEVICE_OPENCL_H */

