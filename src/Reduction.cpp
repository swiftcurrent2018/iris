#include <brisbane/brisbane.h>
#include "Reduction.h"
#include "Debug.h"
#include "Mem.h"

namespace brisbane {
namespace rt {

Reduction::Reduction() {
}

Reduction::~Reduction() {
}


void Reduction::Reduce(Mem* mem, void* host, size_t size) {
    int mode = mem->mode();
    switch (mode) {
        case brisbane_sum: return Sum(mem, host, size);
    }
    _error("not support mode[0x%x]", mode);
}

void Reduction::Sum(Mem* mem, void* host, size_t size) {
    int type = mem->type();
    if (type == brisbane_long) return SumLong(mem, host, size);
    _error("not support type[0x%x]", type);
}

void Reduction::SumLong(Mem* mem, void* host, size_t size) {
    long* src = (long*) mem->host_inter();
    long sum = 0;
    for (int i = 0; i < mem->expansion(); i++) sum += src[i]; 
    if (size != sizeof(sum)) _error("size[%lu] sizeof(sum[%lu])", size, sizeof(sum));
    memcpy(host, &sum, size);
}

Reduction* Reduction::singleton_ = NULL;

Reduction* Reduction::GetInstance() {
    if (singleton_ == NULL) singleton_ = new Reduction();
    return singleton_;
}

} /* namespace rt */
} /* namespace brisbane */
