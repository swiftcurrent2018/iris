#include "LoaderOpenCL.h"
#include "Debug.h"

namespace brisbane {
namespace rt {

LoaderOpenCL::LoaderOpenCL() {
}

LoaderOpenCL::~LoaderOpenCL() {
}

int LoaderOpenCL::LoadFunctions() {
  LOADFUNC(clGetPlatformIDs);
  LOADFUNC(clGetPlatformInfo);
  LOADFUNC(clGetDeviceIDs);
  LOADFUNC(clGetDeviceInfo);
  LOADFUNC(clCreateContext);
  LOADFUNC(clCreateBuffer);
  LOADFUNC(clReleaseMemObject);
  LOADFUNC(clCreateProgramWithSource);
  LOADFUNC(clCreateProgramWithBinary);
  LOADFUNC(clBuildProgram);
  LOADFUNC(clGetProgramBuildInfo);
  LOADFUNC(clCreateKernel);
  LOADFUNC(clSetKernelArg);
  LOADFUNC(clFinish);
  LOADFUNC(clEnqueueReadBuffer);
  LOADFUNC(clEnqueueWriteBuffer);
  LOADFUNC(clEnqueueNDRangeKernel);
  LOADFUNC(clCreateCommandQueue);

  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */

