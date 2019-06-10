#include "Mem.h"
#include "Debug.h"
#include "Device.h"
#include <stdlib.h>

namespace brisbane {
namespace rt {

Mem::Mem(size_t size, Platform* platform) {
    size_ = size;
    platform_ = platform;
    ndevs_ = platform->ndevs();
    host_inter_ = NULL;
    for (int i = 0; i < ndevs_; i++) clmems_[i] = NULL;
    for (int i = 0; i < ndevs_; i++) MemRangeInit(ranges_ + i);
}

Mem::~Mem() {
    _todo("release clmems[%d]", ndevs_);
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

void Mem::SetOwner(Device* dev) {
    AddOwner(0, size_, dev);
}

bool Mem::IsOwner(Device* dev) {
    return ranges_[dev->dev_no()].size != 0;
}

Device* Mem::owner() {
    for (int i = 0; i < ndevs_; i++) {
        if (ranges_[i].size != 0) return platform_->device(i);
    }
    return NULL;
}

void Mem::AddOwner(size_t off, size_t size, Device* dev) {
    int dev_no = dev->dev_no();    
    MemRange* last = ranges_ + dev_no;
    while (last->next != NULL) last = last->next;
    last->off = off;
    last->size = size;
    MemRange* next = new MemRange;
    MemRangeInit(next);
    last->next = next;
}

bool Mem::IsOwner(size_t off, size_t size, Device* dev) {
    int dev_no = dev->dev_no();    
    for (MemRange* range = ranges_ + dev_no; range->next != NULL; range = range->next) {
        if (range->off == off && range->size == size) return true; 
    }
    return false;
}

void Mem::MemRangeInit(MemRange* range) {
    range->off = 0;
    range->size = 0;
    range->next = NULL;
}

} /* namespace rt */
} /* namespace brisbane */
