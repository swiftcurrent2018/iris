#ifndef BRISBANE_RT_SRC_MEM_H
#define BRISBANE_RT_SRC_MEM_H

#include "Retainable.h"
#include "Platform.h"
#include "MemRange.h"
#include <pthread.h>
#include <set>

namespace brisbane {
namespace rt {

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

  cl_mem clmem(int i, cl_context clctx);

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
  cl_mem clmems_[BRISBANE_MAX_NDEVS];
  std::set<MemRange*> ranges_;
  cl_int clerr_;
  void* host_inter_;
  int ndevs_;
  int type_;
  int type_size_;
  int expansion_;

  pthread_mutex_t mutex_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_MEM_H */
