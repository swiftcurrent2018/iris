#include "Mem.h"
#include "Debug.h"
#include "Device.h"
#include <stdlib.h>

namespace brisbane {
namespace rt {

Mem::Mem(size_t size) {
    size_ = size;
    for (int i = 0; i < BRISBANE_MAX_NDEVS; i++) clmems_[i] = NULL;
    owners_[0] = NULL;
    owners_num_ = 0;
    host_inter_ = NULL;
}

Mem::~Mem() {
    _todo("release clmems[%d]", BRISBANE_MAX_NDEVS);
    if (!host_inter_) free(host_inter_);
}

cl_mem Mem::clmem(int i, cl_context clctx) {
    if (clmems_[i] == NULL) {
        clmems_[i] = clCreateBuffer(clctx, CL_MEM_READ_WRITE, size_, NULL, &clerr_);
        _clerror(clerr_);
    }
    return clmems_[i];
}

void* Mem::host_inter() {
    if (!host_inter_) {
        host_inter_ = malloc(size_);
    }
    return host_inter_;
}

void Mem::AddOwner(Device* dev) {
    for (int i = 0; i < owners_num_; i++) {
        if (owners_[i] == dev) return;
    }
    owners_[owners_num_++] = dev;
}

void Mem::SetOwner(Device* dev) {
    owners_num_ = 0;
    AddOwner(dev);
}

bool Mem::IsOwner(Device* dev) {
    if (owners_num_ == 0) return true;
    for (int i = 0; i < owners_num_; i++) {
        if (owners_[i] == dev) return true;
    }
    return false;
}

} /* namespace rt */
} /* namespace brisbane */
