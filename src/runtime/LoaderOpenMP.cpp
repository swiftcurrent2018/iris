#include "LoaderOpenMP.h"
#include "Debug.h"
#include <stdlib.h>

namespace brisbane {
namespace rt {

LoaderOpenMP::LoaderOpenMP() {
}

LoaderOpenMP::~LoaderOpenMP() {
}

int LoaderOpenMP::LoadFunctions() {
  LOADFUNC(brisbane_openmp_init);
  LOADFUNC(brisbane_openmp_finalize);
  LOADFUNC(brisbane_openmp_kernel);
  LOADFUNC(brisbane_openmp_setarg);
  LOADFUNC(brisbane_openmp_setmem);
  LOADFUNC(brisbane_openmp_launch);
  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */

