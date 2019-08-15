#ifndef BRISBANE_RT_SRC_KERNEL_H
#define BRISBANE_RT_SRC_KERNEL_H

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

  cl_kernel clkernel(int i, cl_program clprog);

  std::map<int, KernelArg*>* args() { return &args_; }
  char* name() { return name_; }

  Platform* platform() { return platform_; }
  History* history() { return history_; }

private:
  char name_[256];
  std::map<int, KernelArg*> args_;
  cl_kernel clkernels_[BRISBANE_MAX_NDEVS];

  Platform* platform_;
  History* history_;

  cl_int clerr_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_KERNEL_H */
