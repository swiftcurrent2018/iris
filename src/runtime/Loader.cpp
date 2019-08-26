#include "Loader.h"
#include "Debug.h"

namespace brisbane {
namespace rt {

Loader::Loader() {
  handle_ = NULL;
  handle_ext_ = NULL;
}

Loader::~Loader() {
  if (handle_) if (dlclose(handle_) != 0) _error("%s", dlerror());
  if (handle_ext_) if (dlclose(handle_ext_) != 0) _error("%s", dlerror());
}

int Loader::Load() {
  handle_ = dlopen(library(), RTLD_LAZY);
  if (!handle_ && !library_ext()) return BRISBANE_ERR;
  if (library_ext()) {
    handle_ext_ = dlopen(library_ext(), RTLD_LAZY);
    if (!handle_ext_) return BRISBANE_ERR;
  }
  return LoadFunctions();
}

} /* namespace rt */
} /* namespace brisbane */

