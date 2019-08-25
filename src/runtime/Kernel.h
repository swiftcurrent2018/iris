#ifndef BRISBANE_SRC_RT_KERNEL_H
#define BRISBANE_SRC_RT_KERNEL_H

#include "Config.h"
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

  void** archs() { return archs_; }
  void* arch(Device* dev);

private:
  char name_[256];
  std::map<int, KernelArg*> args_;
  void* archs_[BRISBANE_MAX_NDEVS];
  Device* archs_devs_[BRISBANE_MAX_NDEVS];

  Platform* platform_;
  History* history_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_KERNEL_H */
