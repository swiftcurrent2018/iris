#include "DeviceOpenMP.h"
#include "Debug.h"
#include "Kernel.h"
#include "Mem.h"
#include "Utils.h"
#include <dlfcn.h>

namespace brisbane {
namespace rt {

DeviceOpenMP::DeviceOpenMP(int devno, int platform) : Device(devno, platform) {
  type_ = brisbane_cpu;
  handle_ = NULL;
  FILE* fd = fopen("/proc/cpuinfo", "rb");
  char* arg = 0;
  size_t size = 0;
  while (getdelim(&arg, &size, 0, fd) != -1) {
    char* c1 = strstr(arg, "model name\t: ");
    char* c2 = c1 + strlen("model name\t: ");
    char* c3 = strstr(c2, "GHz");
    _debug("%p", c3);
    if (c1 != 0) {
      strncpy(name_, c2, c3 - c2 + 3);
      break;
    }
  }
  free(arg);
  fclose(fd);
}

DeviceOpenMP::~DeviceOpenMP() {
  if (handle_) {
    dlerr_ = dlclose(handle_);
    if (dlerr_ != 0) _error("%s", dlerror());
  }
}

int DeviceOpenMP::Init() {
  handle_ = dlopen("kernel.omp.so", RTLD_LAZY);
  if (!handle_) {
    _error("%s", dlerror());
    return BRISBANE_ERR;
  }
  *(void**) (&kernel_) = dlsym(handle_, "brisbane_omp_kernel");
  if (!kernel_) _error("%s", dlerror());
  *(void**) (&setarg_) = dlsym(handle_, "brisbane_omp_setarg");
  if (!kernel_) _error("%s", dlerror());
  *(void**) (&setmem_) = dlsym(handle_, "brisbane_omp_setmem");
  if (!kernel_) _error("%s", dlerror());
  *(void**) (&launch_) = dlsym(handle_, "brisbane_omp_launch");
  if (!kernel_) _error("%s", dlerror());
  return BRISBANE_OK;
}

int DeviceOpenMP::H2D(Mem* mem, size_t off, size_t size, void* host) {
  void* mpmem = mem->mpmem(devno_);
  memcpy((char*) mpmem + off, host, size);
  return BRISBANE_OK;
}

int DeviceOpenMP::D2H(Mem* mem, size_t off, size_t size, void* host) {
  void* mpmem = mem->mpmem(devno_);
  memcpy(host, (char*) mpmem + off, size);
  return BRISBANE_OK;
}

int DeviceOpenMP::KernelLaunchInit(Kernel* kernel) {
  if (!kernel_) return BRISBANE_ERR;
  return kernel_(kernel->name());
}

int DeviceOpenMP::KernelSetArg(Kernel* kernel, int idx, size_t size, void* value) {
  if (!setarg_) return BRISBANE_ERR;
  return setarg_(idx, size, value);
}

int DeviceOpenMP::KernelSetMem(Kernel* kernel, int idx, Mem* mem) {
  if (!setmem_) return BRISBANE_ERR;
  void* mpmem = mem->mpmem(devno_);
  return setmem_(idx, mpmem);
}

int DeviceOpenMP::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  _debug("off[%lu] gws[%lu]", off[0], gws[0]);
  if (!launch_) return BRISBANE_ERR;
  return launch_(dim, off[0], gws[0]);
}

} /* namespace rt */
} /* namespace brisbane */

