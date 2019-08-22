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
#ifdef USE_CUDA
  for (int i = 0; i < ndevs_; i++) cumems_[i] = 0;
#endif
#ifdef USE_OPENCL
  for (int i = 0; i < ndevs_; i++) clmems_[i] = NULL;
#endif
  pthread_mutex_init(&mutex_, NULL);
}

Mem::~Mem() {
  for (int i = 0; i < ndevs_; i++) {
#ifdef USE_CUDA
    if (!cumems_[i]) continue;
    cuerr_ = cuMemFree(cumems_[i]);
    _cuerror(cuerr_);
#endif
#ifdef USE_OPENCL
    if (!clmems_[i]) continue;
    clerr_ = clReleaseMemObject(clmems_[i]);
    _clerror(clerr_);
#endif
  }
  if (!host_inter_) free(host_inter_);
  pthread_mutex_destroy(&mutex_);
}

#ifdef USE_CUDA
CUdeviceptr Mem::cumem(int devno) {
  if (cumems_[devno] == 0) {
    cuerr_ = cuMemAlloc(cumems_ + devno, expansion_ * size_);
    _cuerror(cuerr_);
  }
  return cumems_[devno];
}
#endif

#ifdef USE_OPENCL
cl_mem Mem::clmem(int platform, cl_context clctx) {
  if (clmems_[platform] == NULL) {
    clmems_[platform] = clCreateBuffer(clctx, CL_MEM_READ_WRITE, expansion_ * size_, NULL, &clerr_);
    _clerror(clerr_);
  }
  return clmems_[platform];
}
#endif

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

