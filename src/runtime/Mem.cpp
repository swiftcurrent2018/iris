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
  for (int i = 0; i < ndevs_; i++) ranges_[i] = new MemRangeSet();
}

Mem::~Mem() {
  for (int i = 0; i < ndevs_; i++) {
    if (!clmems_[i]) continue;
    clerr_ = clReleaseMemObject(clmems_[i]);
    _clerror(clerr_);
  }
  for (int i = 0; i < ndevs_; i++) delete ranges_[i];
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
  for (int i = 0; i < ndevs_; i++) ranges_[i]->Clear();
  nowners_ = 0;
  AddOwner(0, size_, dev);
}

bool Mem::EmptyOwner() {
  return nowners_ == 0;
}

Device* Mem::owner() {
  for (int i = 0; i < ndevs_; i++) {
    if (!ranges_[i]->Empty()) return platform_->device(i);
  }
  return NULL;
}

void Mem::AddOwner(size_t off, size_t size, Device* dev) {
  int devno = dev->devno();    
  MemRangeSet* range = ranges_[devno];
  if (range->Empty()) nowners_++;
  range->Add(off, size);
}

bool Mem::IsOwner(size_t off, size_t size, Device* dev) {
  int devno = dev->devno();    
  MemRangeSet* range = ranges_[devno];
  return range->Contain(off, size);
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
