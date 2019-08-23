#ifndef BRISBANE_SRC_RT_KERNEL_H
#define BRISBANE_SRC_RT_KERNEL_H

#include "Headers.h"
#include "Retainable.h"
#include "Platform.h"
#include <map>

namespace brisbane {
namespace rt {

class History;

typedef struct _KernelArg {
  size_t size;
  char value[256];
  Mem* mem;
  int mode;
} KernelArg;

class Kernel: public Retainable<struct _brisbane_kernel, Kernel> {
public:
  Kernel(const char* name, Platform* platform);
  virtual ~Kernel();

  int SetArg(int idx, size_t size, void* value);
  int SetMem(int idx, Mem* mem, int mode);
  std::map<int, KernelArg*>* ExportArgs();

  int nargs() { return (int) args_.size(); }
  std::map<int, KernelArg*>* args() { return &args_; }
  char* name() { return name_; }

  Platform* platform() { return platform_; }
  History* history() { return history_; }

private:
  char name_[256];
  std::map<int, KernelArg*> args_;

  Platform* platform_;
  History* history_;

#ifdef USE_CUDA
public:
  CUfunction cukernel(int devno, CUmodule module);
private:
  CUfunction cukernels_[BRISBANE_MAX_NDEVS];
  CUresult cuerr_;
#endif

#ifdef USE_HIP
public:
  hipFunction_t hipkernel(int devno, hipModule_t module);
private:
  hipFunction_t hipkernels_[BRISBANE_MAX_NDEVS];
  hipError_t hiperr_;
#endif

#ifdef USE_OPENCL
public:
  cl_kernel clkernel(int i, cl_program clprog);
private:
  cl_kernel clkernels_[BRISBANE_MAX_NDEVS];
  cl_int clerr_;
#endif
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_KERNEL_H */
