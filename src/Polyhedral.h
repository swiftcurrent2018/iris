#ifndef BRISBANE_RT_SRC_POLYHEDRAL_H
#define BRISBANE_RT_SRC_POLYHEDRAL_H

#include <brisbane/brisbane.h>
#include <brisbane/brisbane_internal.h>

namespace brisbane {
namespace rt {

class Platform;

class Polyhedral {
public:
  Polyhedral(Platform* platform);
  ~Polyhedral();
  int Load();
  int Kernel(const char* name);
  int SetArg(int idx, size_t size, void* value);
  int Launch(int dim, size_t* off, size_t* ndr);
  int GetMem(int idx, brisbane_poly_mem* plmem);

private:
  Platform* platform_;
  void* handle_;
  int dlerr_;

  int (*kernel_)(const char* name);
  int (*setarg_)(int idx, size_t size, void* value);
  int (*launch_)(int dim, size_t* off, size_t* ndr);
  int (*getmem_)(int idx, brisbane_poly_mem* plmem);

  int (*init_)();
  int (*finalize_)();
  int (*lock_)();
  int (*unlock_)();
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_POLYHEDRAL_H */
