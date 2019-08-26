#ifndef BRISBANE_SRC_RT_LOADER_H
#define BRISBANE_SRC_RT_LOADER_H

#include <brisbane/brisbane.h>
#include <dlfcn.h>

#define LOADFUNC(FUNC)      *(void**) (&FUNC) = dlsym(handle_, #FUNC); \
                            if (!FUNC) _error("%s", dlerror())
#define LOADFUNCEXT(FUNC)   *(void**) (&FUNC) = dlsym(handle_ext_, #FUNC); \
                            if (!FUNC) _error("%s", dlerror())

namespace brisbane {
namespace rt {

class Loader {
public:
  Loader();
  virtual ~Loader();

  int Load();
  virtual const char* library() = 0;
  virtual const char* library_ext() { return NULL; }
  virtual int LoadFunctions() = 0;

protected:
  void* handle_;
  void* handle_ext_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_LOADER_H */

