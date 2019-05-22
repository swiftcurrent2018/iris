#include "Mem.h"
#include "Debug.h"

namespace brisbane {
namespace rt {

Mem::Mem(size_t size) {
    _check();
    size_ = size;
    _trace("size[%lu]", size_);
}

Mem::~Mem() {
    _check();
}

} /* namespace rt */
} /* namespace brisbane */
