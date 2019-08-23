#ifndef BRISBANE_INCLUDE_BRISBANE_OMP_H
#define BRISBANE_INCLUDE_BRISBANE_OMP_H

#include <brisbane/brisbane_errno.h>
#include <stdlib.h>
#include <string.h>

#define BRISBANE_OMP_KERNEL_BEGIN   for (_id = _off; _id < _off + _ndr; _id++) {
#define BRISBANE_OMP_KERNEL_END     }
#define BRISBANE_OMP_KERNEL_ARGS    size_t _off, size_t _ndr

#endif /* BRISBANE_INCLUDE_BRISBANE_OMP_H */

