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
  host_inter_ = NULL;
  for (int i = 0; i < ndevs_; i++) clmems_[i] = NULL;
}

Mem::~Mem() {
  for (int i = 0; i < ndevs_; i++) {
    if (!clmems_[i]) continue;
    clerr_ = clReleaseMemObject(clmems_[i]);
    _clerror(clerr_);
  }
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

void Mem::SetOwner(size_t off, size_t size, Device* dev) {
  for (std::set<MemRange*>::iterator I = ranges_.begin(), E = ranges_.end(); I != E;) {
    MemRange* r = *I;
    if (r->Overlap(off, size)) {
      _todo("old[%lu,%lu,%d] new[%lu,%lu,%d]", r->off(), r->size(), r->dev()->devno(), off, size, dev->devno());
      ranges_.erase(I);
      I = ranges_.begin();
    } else ++I;
  }
  ranges_.insert(new MemRange(off, size, dev));
}

void Mem::SetOwner(Device* dev) {
  return SetOwner(0, size_, dev);
}

bool Mem::EmptyOwner() {
  return ranges_.empty();
}

Device* Mem::Owner(size_t off, size_t size) {
  for (std::set<MemRange*>::iterator I = ranges_.begin(), E = ranges_.end(); I != E; ++I) {
    MemRange* r = *I;
    if (r->Contain(off, size)) {
      _debug("dev[%d]", r->dev()->devno());
      return r->dev();
    }
  }
  return NULL;
}

Device* Mem::Owner() {
  return Owner(0, size_);
}

void Mem::AddOwner(size_t off, size_t size, Device* dev) {
  for (std::set<MemRange*>::iterator I = ranges_.begin(), E = ranges_.end(); I != E; ++I) {
    MemRange* r = *I;
    if (r->Overlap(off, size)) {
      _todo("old[%lu,%lu,%d] new[%lu,%lu,%d]", r->off(), r->size(), r->dev()->devno(), off, size, dev->devno());
    }
  }
  ranges_.insert(new MemRange(off, size, dev));
}

bool Mem::IsOwner(size_t off, size_t size, Device* dev) {
  for (std::set<MemRange*>::iterator I = ranges_.begin(), E = ranges_.end(); I != E; ++I) {
    MemRange* r = *I;
    if (r->Contain(off, size) && r->dev() == dev) {
      _check();
      return true;
    }
  }
  return false;
}

bool Mem::IsOwner(Device* dev) {
  return IsOwner(0, size_, dev);
}

void Mem::Reduce(int mode, int type) {
  mode_ = mode;
  type_ = type;
  switch (type_) {
    case brisbane_int:      type_size_ = sizeof(int);       break;
    case brisbane_long:     type_size_ = sizeof(long);      break;
    case brisbane_float:    type_size_ = sizeof(float);     break;
    case brisbane_double:   type_size_ = sizeof(double);    break;
    default: _error("not support type[0x%x]", type_);
  }
}

void Mem::Expand(int expansion) {
  expansion_ = expansion;
}

} /* namespace rt */
} /* namespace brisbane */
