#ifndef BRISBANE_SRC_RT_MEM_H
#define BRISBANE_SRC_RT_MEM_H

#include "Headers.h"
#include "Retainable.h"
#include "MemRange.h"
#include <pthread.h>
#include <set>

namespace brisbane {
namespace rt {

class Platform;

class Mem: public Retainable<struct _brisbane_mem, Mem> {
public:
  Mem(size_t size, Platform* platform);
  virtual ~Mem();

  bool EmptyOwner();
  void AddOwner(size_t off, size_t size, Device* dev);
  void SetOwner(size_t off, size_t size, Device* dev);
  void SetOwner(Device* dev);
  bool IsOwner(size_t off, size_t size, Device* dev);
  bool IsOwner(Device* dev);
  Device* Owner(size_t off, size_t size);
  Device* Owner();
  void Reduce(int mode, int type);
  void Expand(int expansion);

  size_t size() { return size_; }
  int mode() { return mode_; }
  int type() { return type_; }
  int type_size() { return type_size_; }
  int expansion() { return expansion_; }
  void* host_inter();

private:
  size_t size_;
  int mode_;
  Platform* platform_;
  std::set<MemRange*> ranges_;
  void* host_inter_;
  int ndevs_;
  int type_;
  int type_size_;
  int expansion_;

  pthread_mutex_t mutex_;

#ifdef USE_CUDA
public:
  CUdeviceptr* cumems() { return cumems_; }
  CUdeviceptr cumem(int devno);
private:
  CUdeviceptr cumems_[BRISBANE_MAX_NDEVS];
  CUresult cuerr_;
#endif

#ifdef USE_HIP
public:
  void** hipmems() { return hipmems_; }
  void* hipmem(int devno);
private:
  void* hipmems_[BRISBANE_MAX_NDEVS];
  hipError_t hiperr_;
#endif

#ifdef USE_OPENCL
public:
  cl_mem clmem(int platform, cl_context clctx);
private:
  cl_mem clmems_[BRISBANE_MAX_NDEVS];
  cl_int clerr_;
#endif
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_MEM_H */
