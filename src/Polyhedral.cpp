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
    handle_ = dlopen("./libbrisbane_poly.so", RTLD_LAZY);
    if (!handle_) _error("%s", dlerror());

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
    *(void**) (&lock_) = dlsym(handle_, "brisbane_poly_lock");
    if (!lock_) _error("%s", dlerror());
    *(void**) (&unlock_) = dlsym(handle_, "brisbane_poly_unlock");
    if (!unlock_) _error("%s", dlerror());

    init_();
}

} /* namespace rt */
} /* namespace brisbane */

