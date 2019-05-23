#include "Mem.h"
#include "Debug.h"

namespace brisbane {
namespace rt {

Mem::Mem(size_t size) {
    size_ = size;
    for (int i = 0; i < 16; i++) clmems_[i] = NULL;
}

Mem::~Mem() {
    _todo("release clmems[%d]", 16);
}

cl_mem Mem::clmem(int i, cl_context clctx) {
    if (clmems_[i] == NULL) clmems_[i] = clCreateBuffer(clctx, CL_MEM_READ_WRITE, size_, NULL, &clerr_);
    return clmems_[i];
}

} /* namespace rt */
} /* namespace brisbane */
