#ifndef BRISBANE_SRC_RT_PLATFORM_H
#define BRISBANE_SRC_RT_PLATFORM_H

#include <brisbane/brisbane.h>
#include <pthread.h>
#include <stddef.h>
#include <set>
#include <mutex>
#include "Config.h"

namespace brisbane {
namespace rt {

class Device;
class Filter;
class Kernel;
class LoaderCUDA;
class LoaderHIP;
class LoaderOpenCL;
class LoaderOpenMP;
class Mem;
class Polyhedral;
class Profiler;
class Scheduler;
class Task;
class Timer;

class Platform {
private:
  Platform();
  ~Platform();

public:
  int Init(int* argc, char*** argv, int sync);
  int Finalize();
  int Synchronize();

  int InitCUDA();
  int InitHIP();
  int InitOpenCL();
  int InitOpenMP();
  int InitDevices(bool sync);

  int PlatformCount(int* nplatforms);
  int PlatformInfo(int platform, int param, void* value, size_t* size);

  int DeviceCount(int* ndevs);
  int DeviceInfo(int device, int param, void* value, size_t* size);
  int DeviceSetDefault(int device);
  int DeviceGetDefault(int* device);

  int PolicyRegister(const char* lib, const char* name);

  int KernelCreate(const char* name, brisbane_kernel* brs_kernel);
  int KernelSetArg(brisbane_kernel kernel, int idx, size_t size, void* value);
  int KernelSetMem(brisbane_kernel kernel, int idx, brisbane_mem mem, int mode);
  int KernelRelease(brisbane_kernel kernel);

  int TaskCreate(const char* name, brisbane_task* brs_task);
  int TaskDepend(brisbane_task task, int ntasks, brisbane_task* tasks);
  int TaskKernel(brisbane_task brs_task, brisbane_kernel brs_kernel, int dim, size_t* off, size_t* ndr);
  int TaskH2D(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host);
  int TaskD2H(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host);
  int TaskH2DFull(brisbane_task brs_task, brisbane_mem brs_mem, void* host);
  int TaskD2HFull(brisbane_task brs_task, brisbane_mem brs_mem, void* host);
  int TaskSubmit(brisbane_task brs_task, int brs_policy, const char* opt, int wait);
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
  Device* device(int devno) { return devices_[devno]; }
  Polyhedral* polyhedral() { return polyhedral_; }
  Scheduler* scheduler() { return scheduler_; }
  Timer* timer() { return timer_; }
  Kernel* null_kernel() { return null_kernel_; }
  char* app() { return app_; }
  char* host() { return host_; }
  Profiler** profilers() { return profilers_; }
  int nprofilers() { return nprofilers_; }
  double time_app() { return time_app_; }
  double time_init() { return time_init_; }
  bool enable_profiler() { return enable_profiler_; }

private:
  int FilterSubmitExecute(Task* task);
  int ShowKernelHistory();

public:
  static Platform* GetPlatform();

private:
  bool init_;
  bool finalize_;

  char platform_names_[BRISBANE_MAX_NPLATFORMS][64];
  int nplatforms_;
  Device* devices_[BRISBANE_MAX_NDEVS];
  int ndevs_;
  int device_default_;

  LoaderCUDA* loaderCUDA_;
  LoaderHIP* loaderHIP_;
  LoaderOpenCL* loaderOpenCL_;
  LoaderOpenMP* loaderOpenMP_;
  size_t arch_available_;

  std::set<Kernel*> kernels_;
  std::set<Mem*> mems_;

  Scheduler* scheduler_;
  Timer* timer_;
  Polyhedral* polyhedral_;
  bool polyhedral_available_;
  Filter* filter_task_split_;
  bool enable_profiler_;
  Profiler* profilers_[8];
  int nprofilers_;

  Kernel* null_kernel_;

  pthread_mutex_t mutex_;

  char app_[256];
  char host_[256];
  double time_app_;
  double time_init_;

private:
  static Platform* singleton_;
  static std::once_flag flag_singleton_;
  static std::once_flag flag_finalize_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_PLATFORM_H */
