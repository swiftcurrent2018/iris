#ifndef BRISBANE_COMPILER_INCLUDE_BRISBANE_POLY_KERNEL_H
#define BRISBANE_COMPILER_INCLUDE_BRISBANE_POLY_KERNEL_H

#include <stdlib.h>
#define BRISBANE_POLY_KERNEL_BEGIN  for (size_t __gz = __groupoff2; __gz < __ngroups2; __gz++) \
                                    for (size_t __gy = __groupoff1; __gy < __ngroups1; __gy++) \
                                    for (size_t __gx = __groupoff0; __gx < __ngroups0; __gx++) \
                                    for (size_t __lz = 0;           __lz < __lws2;     __lz++) \
                                    for (size_t __ly = 0;           __ly < __lws1;     __ly++) \
                                    for (size_t __lx = 0;           __lx < __lws0;     __lx++) {
#define BRISBANE_POLY_KERNEL_END    }
#define BRISBANE_KERNEL_ARGS        size_t __groupoff0, size_t __groupoff1, size_t __groupoff2, \
                                    size_t __ngroups0,  size_t __ngroups1,  size_t __ngroups2,  \
                                    size_t __gws0,      size_t __gws1,      size_t __gws2,      \
                                    size_t __lws0,      size_t __lws1,      size_t __lws2
#define get_global_id(N)            (N == 0 ? __gx * __lws0 + __lx : N == 1 ? __gy * __lws1 +__ly : __gz * __lws2 + __lz)
#define get_group_id(N)             (N == 0 ? __gx    : N == 1 ? __gy   : __gz)
#define get_local_id(N)             (N == 0 ? __lx    : N == 1 ? __ly   : __lz)
#define get_global_size(N)          (N == 0 ? __gws0  : N == 1 ? __gws1 : __gws2)
#define get_local_size(N)           (N == 0 ? __lws0  : N == 1 ? __lws1 : __lws2)
#define __kernel                    
#define __global                    
#define __local

#endif /* BRISBANE_COMPILER_INCLUDE_BRISBANE_POLY_KERNEL_H */

