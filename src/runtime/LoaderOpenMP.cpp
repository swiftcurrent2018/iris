#include "LoaderOpenMP.h"
#include "Debug.h"

namespace brisbane {
namespace rt {

LoaderOpenMP::LoaderOpenMP() {

}

LoaderOpenMP::~LoaderOpenMP() {

}

int LoaderOpenMP::LoadFunctions() {
  if (!handle_) return BRISBANE_ERR;

  LOADFUNC(omp_get_num_procs);
  LOADFUNC(omp_get_max_threads);

  LOADFUNCEXT(brisbane_openmp_init);
  LOADFUNCEXT(brisbane_openmp_finalize);
  LOADFUNCEXT(brisbane_openmp_kernel);
  LOADFUNCEXT(brisbane_openmp_setarg);
  LOADFUNCEXT(brisbane_openmp_setmem);
  LOADFUNCEXT(brisbane_openmp_launch);

  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */

