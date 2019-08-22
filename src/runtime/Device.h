#ifndef BRISBANE_SRC_RT_DEVICE_H
#define BRISBANE_SRC_RT_DEVICE_H

#include "Headers.h"

namespace brisbane {
namespace rt {

class Command;
class Kernel;
class Mem;
class Task;
class Timer;
class Worker;

class Device {
public:
  Device(int devno, int platform);
  virtual ~Device();

  void Execute(Task* task);
  void ExecuteInit(Command* cmd);
  void ExecuteKernel(Command* cmd);
  void ExecuteH2D(Command* cmd);
  void ExecuteH2DNP(Command* cmd);
  void ExecuteD2H(Command* cmd);
  void ExecuteReleaseMem(Command* cmd);

  virtual int Init() = 0;
  virtual int H2D(Mem* mem, size_t off, size_t size, void* host) = 0;
  virtual int D2H(Mem* mem, size_t off, size_t size, void* host) = 0;
  virtual int KernelSetArg(Kernel* kernel, int idx, size_t arg_size, void* arg_value) = 0;
  virtual int KernelSetMem(Kernel* kernel, int idx, Mem* mem) = 0;
  virtual int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) = 0;

  int devno() { return devno_; }
  int type() { return type_; }
  char* name() { return name_; }
  bool busy() { return busy_; }
  bool idle() { return !busy_; }
  bool enable() { return enable_; }

  void set_worker(Worker* worker) { worker_ = worker; }
  Worker* worker() { return worker_; }

protected:
  int devno_;
  int platform_;
  int type_;
  char vendor_[64];
  char name_[64];
  char version_[64];
  int max_compute_units_;
  size_t max_work_item_sizes_[3];

  bool busy_;
  bool enable_;

  Worker* worker_;
  Timer* timer_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_DEVICE_H */
