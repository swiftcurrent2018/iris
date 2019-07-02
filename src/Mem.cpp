#include "Mem.h"
#include "Debug.h"
#include "Device.h"
#include <stdlib.h>

namespace brisbane {
namespace rt {

Mem::Mem(size_t size, Platform* platform) {
    size_ = size;
    mode_ = brisbane_normal;
    expansion_ = 1;
    platform_ = platform;
    ndevs_ = platform->ndevs();
    nowners_ = 0;
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
        clmems_[i] = clCreateBuffer(clctx, CL_MEM_READ_WRITE, expansion_ * size_, NULL, &clerr_);
        _clerror(clerr_);
    }
    return clmems_[i];
}

void* Mem::host_inter() {
    if (!host_inter_) {
        host_inter_ = malloc(expansion_ * size_);
    }
    return host_inter_;
}

void Mem::SetOwner(Device* dev) {
    if (nowners_ == 1 && IsOwner(dev)) return;
    for (int i = 0; i < ndevs_; i++) {
        MemRange* range = ranges_ + i;    
        range->size = 0UL;
        range->next = NULL;
    }
    nowners_ = 0;
    AddOwner(0, size_, dev);
}

bool Mem::IsOwner(Device* dev) {
    if (nowners_ == 0) return true;
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
    nowners_++;
}

bool Mem::IsOwner(size_t off, size_t size, Device* dev) {
    int dev_no = dev->dev_no();    
    for (MemRange* range = ranges_ + dev_no; range->next != NULL; range = range->next) {
        if (range->off == off && range->size == size) return true; 
    }
    return false;
}

void Mem::Reduce(int mode, int type) {
    mode_ = mode;
    type_ = type;
}

void Mem::Expand(int expansion) {
    expansion_ = expansion;
}

void Mem::MemRangeInit(MemRange* range) {
    range->off = 0;
    range->size = 0;
    range->next = NULL;
}

} /* namespace rt */
} /* namespace brisbane */
