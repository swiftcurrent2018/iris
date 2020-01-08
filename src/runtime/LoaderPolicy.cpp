#include "LoaderPolicy.h"
#include "Policy.h"
#include "Debug.h"

namespace brisbane {
namespace rt {

LoaderPolicy::LoaderPolicy(const char* lib, const char* name) : Loader() {
  memset(lib_, 0, sizeof(lib_));
  memset(name_, 0, sizeof(name_));
  strncpy(lib_, lib, strlen(lib));
  strncpy(name_, name, strlen(name));
}

LoaderPolicy::~LoaderPolicy() {
}

Policy* LoaderPolicy::policy() {
  return (Policy*) (instance_)();
}

const char* LoaderPolicy::library() {
  return lib_;
}

int LoaderPolicy::LoadFunctions() {
  char func[128];
  sprintf(func, "%s_instance", name_);
  *(void**) (&instance_) = dlsym(handle_, func);
  if (!instance_) {
    _error("%s", dlerror());
    return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */

