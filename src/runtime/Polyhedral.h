#ifndef BRISBANE_SRC_RT_POLYHEDRAL_H
#define BRISBANE_SRC_RT_POLYHEDRAL_H

#include <brisbane/brisbane.h>
#include <brisbane/brisbane_poly_types.h>

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
  int Launch(int dim, size_t* wgo, size_t* wgs, size_t* gws, size_t* lws);
  int GetMem(int idx, brisbane_poly_mem* plmem);

private:
  Platform* platform_;
  void* handle_;
  int dlerr_;

  int (*kernel_)(const char* name);
  int (*setarg_)(int idx, size_t size, void* value);
  int (*launch_)(int dim, size_t* wgo, size_t* wgs, size_t* gws, size_t* lws);
  int (*getmem_)(int idx, brisbane_poly_mem* plmem);

  int (*init_)();
  int (*finalize_)();
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_POLYHEDRAL_H */
