#include "Mem.h"
#include "Debug.h"
#include "Platform.h"
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
  pthread_mutex_init(&mutex_, NULL);
  for (int i = 0; i < ndevs_; i++) {
    archs_[i] = NULL;
    archs_devs_[i] = NULL;
  }
}

Mem::~Mem() {
  for (int i = 0; i < ndevs_; i++) {
    if (archs_devs_[i]) archs_devs_[i]->MemFree(archs_[i]);
  }
  if (!host_inter_) free(host_inter_);
  pthread_mutex_destroy(&mutex_);
}

void* Mem::arch(Device* dev) {
  int devno = dev->devno();
  if (archs_[devno] == NULL) dev->MemAlloc(archs_ + devno, expansion_ * size_);
  return archs_[devno];
}

void* Mem::host_inter() {
  if (!host_inter_) {
    host_inter_ = malloc(expansion_ * size_);
  }
  return host_inter_;
}

void Mem::AddOwner(size_t off, size_t size, Device* dev) {
  pthread_mutex_lock(&mutex_);
  _trace("mem[%lu] off[%lu] size[%lu] dev[%d]", uid(), off, size, dev->devno());
  for (std::set<MemRange*>::iterator I = ranges_.begin(), E = ranges_.end(); I != E; ++I) {
    MemRange* r = *I;
    if (r->Overlap(off, size)) {
      _trace("old[%lu,%lu,%d] new[%lu,%lu,%d]", r->off(), r->size(), r->dev()->devno(), off, size, dev->devno());
    }
  }
  ranges_.insert(new MemRange(off, size, dev));
  pthread_mutex_unlock(&mutex_);
}

void Mem::SetOwner(size_t off, size_t size, Device* dev) {
  pthread_mutex_lock(&mutex_);
  _trace("mem[%lu] off[%lu] size[%lu] dev[%d]", uid(), off, size, dev->devno());
  for (std::set<MemRange*>::iterator I = ranges_.begin(), E = ranges_.end(); I != E;) {
    MemRange* r = *I;
    if (r->Overlap(off, size)) {
      _todo("old[%lu,%lu,%d] new[%lu,%lu,%d]", r->off(), r->size(), r->dev()->devno(), off, size, dev->devno());
      ranges_.erase(I);
      I = ranges_.begin();
    } else ++I;
  }
  ranges_.insert(new MemRange(off, size, dev));
  pthread_mutex_unlock(&mutex_);
}

void Mem::SetOwner(Device* dev) {
  return SetOwner(0, size_, dev);
}

bool Mem::EmptyOwner() {
  bool bret = true;
  pthread_mutex_lock(&mutex_);
  bret = ranges_.empty();
  pthread_mutex_unlock(&mutex_);
  return bret;
}

Device* Mem::Owner(size_t off, size_t size) {
  pthread_mutex_lock(&mutex_);
  for (std::set<MemRange*>::iterator I = ranges_.begin(), E = ranges_.end(); I != E; ++I) {
    MemRange* r = *I;
    if (r->Contain(off, size)) {
      pthread_mutex_unlock(&mutex_);
      return r->dev();
    }
  }
  pthread_mutex_unlock(&mutex_);
  return NULL;
}

Device* Mem::Owner() {
  return Owner(0, size_);
}

bool Mem::IsOwner(size_t off, size_t size, Device* dev) {
  pthread_mutex_lock(&mutex_);
  for (std::set<MemRange*>::iterator I = ranges_.begin(), E = ranges_.end(); I != E; ++I) {
    MemRange* r = *I;
    if (r->Contain(off, size) && r->dev() == dev) {
      pthread_mutex_unlock(&mutex_);
      return true;
    }
  }
  pthread_mutex_unlock(&mutex_);
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

