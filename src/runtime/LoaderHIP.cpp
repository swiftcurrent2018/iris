#include "LoaderHIP.h"
#include "Debug.h"

namespace brisbane {
namespace rt {

LoaderHIP::LoaderHIP() {
}

LoaderHIP::~LoaderHIP() {
}

int LoaderHIP::LoadFunctions() {
  LOADFUNC(hipInit);
  LOADFUNC(hipDriverGetVersion);
  LOADFUNC(hipGetDeviceCount);
  LOADFUNC(hipDeviceGet);
  LOADFUNC(hipDeviceGetName);
  LOADFUNC(hipCtxCreate);
  LOADFUNC(hipModuleLoad);
  LOADFUNC(hipModuleGetFunction);
  LOADFUNC(hipMalloc);
  LOADFUNC(hipFree);
  LOADFUNC(hipMemcpyHtoD);
  LOADFUNC(hipMemcpyDtoH);
  LOADFUNC(hipModuleLaunchKernel);
  LOADFUNC(hipDeviceSynchronize);

  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */

