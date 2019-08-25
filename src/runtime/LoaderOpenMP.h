#ifndef BRISBANE_SRC_RT_LOADER_OPENMP_H
#define BRISBANE_SRC_RT_LOADER_OPENMP_H

#include "Loader.h"
#include <omp.h>

namespace brisbane {
namespace rt {

class LoaderOpenMP : public Loader {
public:
  LoaderOpenMP();
  ~LoaderOpenMP();

  const char* library() { return "libgomp.so.1"; }
  const char* library_ext() { return "kernel.openmp.so"; }
  int LoadFunctions();

  int (*omp_get_num_procs)(void);
  int (*omp_get_max_threads)(void);

  int (*brisbane_openmp_init)();
  int (*brisbane_openmp_finalize)();
  int (*brisbane_openmp_kernel)(const char* name);
  int (*brisbane_openmp_setarg)(int idx, size_t size, void* value);
  int (*brisbane_openmp_setmem)(int idx, void* mem);
  int (*brisbane_openmp_launch)(int dim, size_t off, size_t ndr);
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_LOADER_OPENMP_H */

