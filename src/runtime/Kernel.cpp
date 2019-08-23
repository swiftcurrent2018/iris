#include "Kernel.h"
#include "Debug.h"
#include "History.h"
#include <string.h>

namespace brisbane {
namespace rt {

Kernel::Kernel(const char* name, Platform* platform) {
  size_t len = strlen(name);
  strncpy(name_, name, len);
  name_[len] = 0;
  platform_ = platform;
  history_ = new History(this);
#ifdef USE_CUDA
  for (int i = 0; i < BRISBANE_MAX_NDEVS; i++) cukernels_[i] = NULL;
#endif
#ifdef USE_HIP
  for (int i = 0; i < BRISBANE_MAX_NDEVS; i++) hipkernels_[i] = NULL;
#endif
#ifdef USE_OPENCL
  for (int i = 0; i < BRISBANE_MAX_NDEVS; i++) clkernels_[i] = NULL;
#endif
}

Kernel::~Kernel() {
  for (std::map<int, KernelArg*>::iterator I = args_.begin(), E = args_.end(); I != E; ++I)
    delete I->second;
}

int Kernel::SetArg(int idx, size_t size, void* value) {
  KernelArg* arg = new KernelArg;
  arg->size = size;
  if (value) memcpy(arg->value, value, size);
  arg->mem = NULL;
  args_[idx] = arg;
  return BRISBANE_OK;
}

int Kernel::SetMem(int idx, Mem* mem, int mode) {
  KernelArg* arg = new KernelArg;
  arg->mem = mem;
  arg->mode = mode;
  args_[idx] = arg;
  return BRISBANE_OK;
}

std::map<int, KernelArg*>* Kernel::ExportArgs() {
  std::map<int, KernelArg*>* new_args = new std::map<int, KernelArg*>();
  for (std::map<int, KernelArg*>::iterator I = args_.begin(), E = args_.end(); I != E; ++I) {
    KernelArg* arg = I->second;
    KernelArg* new_arg = new KernelArg;
    if (arg->mem) {
      new_arg->mem = arg->mem;
      new_arg->mode = arg->mode;
    } else {
      new_arg->size = arg->size; 
      memcpy(new_arg->value, arg->value, arg->size);
      new_arg->mem = NULL;
    }
    (*new_args)[I->first] = new_arg;
  }
  return new_args;
}

#ifdef USE_HIP
hipFunction_t Kernel::hipkernel(int devno, hipModule_t module) {
  if (hipkernels_[devno] == NULL) {
    hiperr_ = hipModuleGetFunction(hipkernels_ + devno, module, (const char*) name_);
    _hiperror(hiperr_);
  }
  return hipkernels_[devno];
}
#endif

#ifdef USE_CUDA
CUfunction Kernel::cukernel(int devno, CUmodule module) {
  if (cukernels_[devno] == NULL) {
    cuerr_ = cuModuleGetFunction(cukernels_ + devno, module, (const char*) name_);
    _cuerror(cuerr_);
  }
  return cukernels_[devno];
}
#endif

#ifdef USE_OPENCL
cl_kernel Kernel::clkernel(int i, cl_program clprog) {
  if (clkernels_[i] == NULL) {
    clkernels_[i] = clCreateKernel(clprog, (const char*) name_, &clerr_);
    _clerror(clerr_);
  }
  return clkernels_[i];
}
#endif

} /* namespace rt */
} /* namespace brisbane */

