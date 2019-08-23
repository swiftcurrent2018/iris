#ifndef BRISBANE_RT_SRC_HEADERS_H
#define BRISBANE_RT_SRC_HEADERS_H

#include "Config.h"

#ifdef USE_CUDA
#include <cuda/cuda.h>
#endif

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#endif

#ifdef USE_OPENCL
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

#endif /* BRISBANE_RT_SRC_HEADERS_H */

