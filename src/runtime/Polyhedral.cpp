#include "Polyhedral.h"
#include "Debug.h"
#include <dlfcn.h>

namespace brisbane {
namespace rt {

Polyhedral::Polyhedral(Platform* platform) {
  platform_ = platform;
  handle_ = NULL;
}

Polyhedral::~Polyhedral() {
  if (handle_) {
    finalize_();

    dlerr_ = dlclose(handle_);
    if (dlerr_ != 0) _error("%s", dlerror());
  }
}

int Polyhedral::Load() {
  handle_ = dlopen("./brisbane/libbrisbane_poly.so", RTLD_LAZY);
  if (!handle_) return BRISBANE_ERR;

  *(void**) (&kernel_) = dlsym(handle_, "brisbane_poly_kernel");
  if (!kernel_) _error("%s", dlerror());
  *(void**) (&setarg_) = dlsym(handle_, "brisbane_poly_setarg");
  if (!kernel_) _error("%s", dlerror());
  *(void**) (&launch_) = dlsym(handle_, "brisbane_poly_launch");
  if (!kernel_) _error("%s", dlerror());
  *(void**) (&getmem_) = dlsym(handle_, "brisbane_poly_getmem");
  if (!kernel_) _error("%s", dlerror());

  *(void**) (&init_) = dlsym(handle_, "brisbane_poly_init");
  if (!init_) _error("%s", dlerror());
  *(void**) (&finalize_) = dlsym(handle_, "brisbane_poly_finalize");
  if (!finalize_) _error("%s", dlerror());

  init_();

  return BRISBANE_OK;
}

int Polyhedral::Kernel(const char* name) {
  if (!kernel_) return BRISBANE_ERR;
  return kernel_(name);
}

int Polyhedral::SetArg(int idx, size_t size, void* value) {
  if (!setarg_) return BRISBANE_ERR;
  return setarg_(idx, size, value);
}

int Polyhedral::Launch(int dim, size_t* wgo, size_t* wgs, size_t* gws, size_t* lws) {
  if (!launch_) return BRISBANE_ERR;
  return launch_(dim, wgo, wgs, gws, lws);
}

int Polyhedral::GetMem(int idx, brisbane_poly_mem* plmem) {
  if (!getmem_) return BRISBANE_ERR;
  return getmem_(idx, plmem);
}

} /* namespace rt */
} /* namespace brisbane */

