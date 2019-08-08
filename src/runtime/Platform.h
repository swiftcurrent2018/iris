#ifndef BRISBANE_RT_SRC_PLATFORM_H
#define BRISBANE_RT_SRC_PLATFORM_H

#define CL_TARGET_OPENCL_VERSION 120

#include <brisbane/brisbane.h>
#include <CL/cl.h>
#include <stddef.h>
#include <set>
#include "Config.h"
#include "Debug.h"

namespace brisbane {
namespace rt {

class Device;
class DOT;
class Filter;
class Kernel;
class Mem;
class Polyhedral;
class Scheduler;
class Task;
class Timer;

class Platform {
private:
  Platform();
  ~Platform();

public:
  int Init(int* argc, char*** argv, bool sync);
  int Synchronize();

  int InitCLPlatforms();
  int BuildPrograms(bool sync);

  int InfoNumPlatforms(int* nplatforms);
  int InfoNumDevices(int* ndevs);

  int DeviceSetDefault(int device);
  int DeviceGetDefault(int* device);

  int KernelCreate(const char* name, brisbane_kernel* brs_kernel);
  int KernelSetArg(brisbane_kernel kernel, int idx, size_t arg_size, void* arg_value);
  int KernelSetMem(brisbane_kernel kernel, int idx, brisbane_mem mem, int mode);
  int KernelRelease(brisbane_kernel kernel);

  int TaskCreate(const char* name, brisbane_task* brs_task);
  int TaskDepend(brisbane_task task, int ntasks, brisbane_task* tasks);
  int TaskKernel(brisbane_task brs_task, brisbane_kernel brs_kernel, int dim, size_t* off, size_t* ndr);
  int TaskH2D(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host);
  int TaskD2H(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host);
  int TaskH2DFull(brisbane_task brs_task, brisbane_mem brs_mem, void* host);
  int TaskD2HFull(brisbane_task brs_task, brisbane_mem brs_mem, void* host);
  int TaskPresent(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host);
  int TaskSubmit(brisbane_task brs_task, int brs_device, char* opt, bool wait);
  int TaskWait(brisbane_task brs_task);
  int TaskWaitAll(int ntasks, brisbane_task* brs_tasks);
  int TaskAddSubtask(brisbane_task brs_task, brisbane_task brs_subtask);
  int TaskRelease(brisbane_task brs_task);
  int TaskReleaseMem(brisbane_task brs_task, brisbane_mem brs_mem);

  int MemCreate(size_t size, brisbane_mem* brs_mem);
  int MemReduce(brisbane_mem brs_mem, int mode, int type);
  int MemRelease(brisbane_mem brs_mem);

  int TimerNow(double* time);

  int ndevs() { return ndevs_; }
  int device_default() { return device_default_; }
  Device** devices() { return devices_; }
  Device* device(int dev_no) { return devices_[dev_no]; }
  Polyhedral* polyhedral() { return polyhedral_; }
  Scheduler* scheduler() { return scheduler_; }
  Timer* timer() { return timer_; }
  Kernel* null_kernel() { return null_kernel_; }
  char* app() { return app_; }
  char* host() { return host_; }
  DOT* dot() { return dot_; }
  double time_app() { return time_app_; }
  double time_init() { return time_init_; }

private:
  int FilterSubmitExecute(Task* task);
  int ShowKernelHistory();

public:
  static Platform* GetPlatform();
  static int Finalize();

private:
  bool init_;

  Device* devices_[BRISBANE_MAX_NDEVS];
  int nplatforms_;
  int ndevs_;
  int device_default_;

  cl_platform_id cl_platforms_[BRISBANE_MAX_NDEVS];
  cl_context cl_contexts_[BRISBANE_MAX_NDEVS];
  cl_device_id cl_devices_[BRISBANE_MAX_NDEVS];
  cl_int clerr_;

  std::set<Kernel*> kernels_;
  std::set<Mem*> mems_;

  Scheduler* scheduler_;
  Timer* timer_;
  Polyhedral* polyhedral_;
  bool polyhedral_available_;
  Filter* filter_task_split_;

  Kernel* null_kernel_;

  DOT* dot_;
  char app_[256];
  char host_[256];
  double time_app_;
  double time_init_;

private:
  static Platform* singleton_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_PLATFORM_H */
