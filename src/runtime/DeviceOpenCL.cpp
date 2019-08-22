#include "DeviceOpenCL.h"
#include "Command.h"
#include "History.h"
#include "Kernel.h"
#include "Mem.h"
#include "Reduction.h"
#include "Timer.h"
#include "Utils.h"

namespace brisbane {
namespace rt {

DeviceOpenCL::DeviceOpenCL(cl_device_id cldev, cl_context clctx, int devno, int platform) : Device(devno, platform) {
  cldev_ = cldev;
  clctx_ = clctx;
  err_ = clGetDeviceInfo(cldev_, CL_DEVICE_VENDOR, sizeof(vendor_), vendor_, NULL);
  err_ = clGetDeviceInfo(cldev_, CL_DEVICE_NAME, sizeof(name_), name_, NULL);
  err_ = clGetDeviceInfo(cldev_, CL_DEVICE_TYPE, sizeof(cltype_), &cltype_, NULL);
  err_ = clGetDeviceInfo(cldev_, CL_DEVICE_VERSION, sizeof(version_), version_, NULL);
  err_ = clGetDeviceInfo(cldev_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units_), &max_compute_units_, NULL);
  err_ = clGetDeviceInfo(cldev_, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes_), max_work_item_sizes_, NULL);
  err_ = clGetDeviceInfo(cldev_, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compiler_available_), &compiler_available_, NULL);

  if (cltype_ == CL_DEVICE_TYPE_CPU) type_ = brisbane_cpu;
  else if (cltype_ == CL_DEVICE_TYPE_GPU) {
    type_ = brisbane_gpu;
    if (strcasestr(vendor_, "NVIDIA")) type_ = brisbane_nvidia;
    else if (strcasestr(vendor_, "AMD")) type_ = brisbane_amd;
  }
  else if (cltype_ == CL_DEVICE_TYPE_ACCELERATOR) {
    if (strstr(name_, "FPGA") != NULL || strstr(version_, "FPGA") != NULL) type_ = brisbane_fpga;
    else type_ = brisbane_phi;
  }
  else type_ = brisbane_cpu;

  _info("device[%d] platform[%d] vendor[%s] device[%s] type[%d] version[%s] max_compute_units[%d] max_work_item_sizes[%lu,%lu,%lu] compiler_available[%d]", devno_, platform_, vendor_, name_, type_, version_, max_compute_units_, max_work_item_sizes_[0], max_work_item_sizes_[1], max_work_item_sizes_[2], compiler_available_);
}

DeviceOpenCL::~DeviceOpenCL() {
}

int DeviceOpenCL::Init() {
  clcmdq_ = clCreateCommandQueue(clctx_, cldev_, 0, &err_);
  _clerror(err_);

  cl_int status;
  char path[256];
  sprintf(path, "kernel-%s",
    type_ == brisbane_cpu    ? "cpu.cl"  :
    type_ == brisbane_nvidia ? "nvidia.cl"  :
    type_ == brisbane_amd    ? "amd.cl"  :
    type_ == brisbane_gpu    ? "gpu.cl"  :
    type_ == brisbane_phi    ? "phi.cl"  :
    type_ == brisbane_fpga   ? "fpga.aocx" : "default.cl");
  char* src = NULL;
  size_t srclen = 0;
  if (Utils::ReadFile(path, &src, &srclen) == BRISBANE_ERR) {
    sprintf(path, "kernel.cl");
    Utils::ReadFile(path, &src, &srclen);
  }
  if (srclen == 0) {
    _error("dev[%d][%s] has no kernel file", devno_, name_);
    return BRISBANE_OK;
  }
  _trace("dev[%d][%s] kernels[%s]", devno_, name_, path);
  if (type_ == brisbane_fpga) clprog_ = clCreateProgramWithBinary(clctx_, 1, &cldev_, (const size_t*) &srclen, (const unsigned char**) &src, &status, &err_);
  else clprog_ = clCreateProgramWithSource(clctx_, 1, (const char**) &src, (const size_t*) &srclen, &err_);
  _clerror(err_);
  err_ = clBuildProgram(clprog_, 1, &cldev_, "", NULL, NULL);
  _clerror(err_);
  if (err_ != CL_SUCCESS) {
    cl_build_status s;
    err_ = clGetProgramBuildInfo(clprog_, cldev_, CL_PROGRAM_BUILD_STATUS, sizeof(s), &s, NULL);
    _clerror(err_);
    char log[1024];
    size_t log_size;
    err_ = clGetProgramBuildInfo(clprog_, cldev_, CL_PROGRAM_BUILD_LOG, 1024, log, &log_size);
    _clerror(err_);
    _error("status[%d] log_size[%lu] log:%s", s, log_size, log);
    _error("srclen[%lu] src\n%s", srclen, src);
    if (src) free(src);
    return BRISBANE_ERR;
  }
  if (src) free(src);

  return BRISBANE_OK;
}

int DeviceOpenCL::H2D(Mem* mem, size_t off, size_t size, void* host) {
  cl_mem clmem = mem->clmem(platform_, clctx_);
  err_ = clEnqueueWriteBuffer(clcmdq_, clmem, CL_TRUE, off, size, host, 0, NULL, NULL);
  _clerror(err_);
  return BRISBANE_OK;
}

int DeviceOpenCL::D2H(Mem* mem, size_t off, size_t size, void* host) {
  cl_mem clmem = mem->clmem(platform_, clctx_);
  err_ = clEnqueueReadBuffer(clcmdq_, clmem, CL_TRUE, off, size, host, 0, NULL, NULL);
  _clerror(err_);
  return BRISBANE_OK;
}

int DeviceOpenCL::KernelSetArg(Kernel* kernel, int idx, size_t arg_size, void* arg_value) {
  cl_kernel clkernel = kernel->clkernel(devno_, clprog_);
  err_ = clSetKernelArg(clkernel, (cl_uint) idx, arg_size, arg_value);
  _clerror(err_);
  return BRISBANE_OK;
}

int DeviceOpenCL::KernelSetMem(Kernel* kernel, int idx, Mem* mem) {
  cl_kernel clkernel = kernel->clkernel(devno_, clprog_);
  cl_mem clmem = mem->clmem(platform_, clctx_);
  err_ = clSetKernelArg(clkernel, (cl_uint) idx, sizeof(clmem), (const void*) &clmem);
  _clerror(err_);
  return BRISBANE_OK;
}

int DeviceOpenCL::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  cl_kernel clkernel = kernel->clkernel(devno_, clprog_);
  err_ = clEnqueueNDRangeKernel(clcmdq_, clkernel, (cl_uint) dim, (const size_t*) off, (const size_t*) gws, (const size_t*) lws, 0, NULL, NULL);
  _clerror(err_);
  err_ = clFinish(clcmdq_);
  _clerror(err_);
  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */

