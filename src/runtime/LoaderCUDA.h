#ifndef BRISBANE_SRC_RT_LOADER_CUDA_H
#define BRISBANE_SRC_RT_LOADER_CUDA_H

#include "Loader.h"
#include <cuda/cuda.h>

namespace brisbane {
namespace rt {

class LoaderCUDA : public Loader {
public:
  LoaderCUDA();
  ~LoaderCUDA();

  const char* library() { return "libcuda.so"; }
  int LoadFunctions();

  CUresult (*cuInit)(unsigned int Flags);
  CUresult (*cuDriverGetVersion)(int* driverVersion);
  CUresult (*cuDeviceGet)(CUdevice* device, int ordinal);
  CUresult (*cuDeviceGetAttribute)(int* pi, CUdevice_attribute attrib, CUdevice dev);
  CUresult (*cuDeviceGetCount)(int* count);
  CUresult (*cuDeviceGetName)(char* name, int len, CUdevice dev);
  CUresult (*cuCtxCreate)(CUcontext* pctx, unsigned int flags,CUdevice dev);
  CUresult (*cuStreamCreate)(CUstream* phStream, unsigned int Flags);
  CUresult (*cuStreamSynchronize)(CUstream hStream);
  CUresult (*cuModuleGetFunction)(CUfunction* hfunc, CUmodule hmod, const char* name);
  CUresult (*cuModuleLoad)(CUmodule* module, const char* fname);
  CUresult (*cuMemAlloc)(CUdeviceptr* dptr, size_t bytesize);
  CUresult (*cuMemFree)(CUdeviceptr dptr);
  CUresult (*cuMemcpyHtoD)(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
  CUresult (*cuMemcpyDtoH)(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
  CUresult (*cuLaunchKernel)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra);
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_LOADER_CUDA_H */

