#include "LoaderCUDA.h"
#include "Debug.h"

namespace brisbane {
namespace rt {

LoaderCUDA::LoaderCUDA() {
}

LoaderCUDA::~LoaderCUDA() {
}

int LoaderCUDA::LoadFunctions() {
  LOADFUNC(cuInit);
  LOADFUNC(cuDriverGetVersion);
  LOADFUNC(cuDeviceGet);
  LOADFUNC(cuDeviceGetAttribute);
  LOADFUNC(cuDeviceGetCount);
  LOADFUNC(cuDeviceGetName);
  LOADFUNC(cuCtxCreate);
  LOADFUNC(cuCtxSynchronize);
  LOADFUNC(cuStreamAddCallback);
  LOADFUNC(cuStreamCreate);
  LOADFUNC(cuStreamSynchronize);
  LOADFUNC(cuModuleGetFunction);
  LOADFUNC(cuModuleLoad);
  LOADFUNC(cuMemAlloc);
  LOADFUNC(cuMemFree);
  LOADFUNC(cuMemcpyHtoD);
  LOADFUNC(cuMemcpyHtoDAsync);
  LOADFUNC(cuMemcpyDtoH);
  LOADFUNC(cuMemcpyDtoHAsync);
//  LOADFUNC(cuLaunchHostFunc);
  LOADFUNC(cuLaunchKernel);
  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */

