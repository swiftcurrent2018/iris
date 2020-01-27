#include "Platform.h"
#include "Debug.h"
#include "Utils.h"
#include "Command.h"
#include "DeviceCUDA.h"
#include "DeviceHIP.h"
#include "DeviceOpenCL.h"
#include "DeviceOpenMP.h"
#include "FilterTaskSplit.h"
#include "History.h"
#include "Kernel.h"
#include "LoaderCUDA.h"
#include "LoaderHIP.h"
#include "LoaderOpenCL.h"
#include "LoaderOpenMP.h"
#include "Mem.h"
#include "Policies.h"
#include "Polyhedral.h"
#include "PresentTable.h"
#include "Profiler.h"
#include "ProfilerDOT.h"
#include "ProfilerGoogleCharts.h"
#include "Scheduler.h"
#include "Task.h"
#include "Timer.h"
#include "Worker.h"
#include <unistd.h>
#include <algorithm>

namespace brisbane {
namespace rt {

char brisbane_log_prefix_[256];

Platform::Platform() {
  init_ = false;
  finalize_ = false;
  nplatforms_ = 0;
  ndevs_ = 0;

  scheduler_ = NULL;
  polyhedral_ = NULL;
  filter_task_split_ = NULL;
  timer_ = NULL;
  null_kernel_ = NULL;
  loaderCUDA_ = NULL;
  loaderHIP_ = NULL;
  loaderOpenCL_ = NULL;
  loaderOpenMP_ = NULL;
  arch_available_ = 0UL;
  present_table_ = NULL;
  enable_profiler_ = getenv("BRISBANE_PROFILE");
  nprofilers_ = 0;
  time_app_ = 0.0;
  time_init_ = 0.0;

  pthread_mutex_init(&mutex_, NULL);
}

Platform::~Platform() {
  if (!init_) return;

  if (scheduler_) delete scheduler_;
  if (loaderCUDA_) delete loaderCUDA_;
  if (loaderHIP_) delete loaderHIP_;
  if (loaderOpenCL_) delete loaderOpenCL_;
  if (loaderOpenMP_) delete loaderOpenMP_;
  if (present_table_) delete present_table_;
  if (polyhedral_) delete polyhedral_;
  if (filter_task_split_) delete filter_task_split_;
  if (timer_) delete timer_;
  if (null_kernel_) delete null_kernel_;
  if (enable_profiler_)
    for (int i = 0; i < nprofilers_; i++) delete profilers_[i];

  pthread_mutex_destroy(&mutex_);
}

int Platform::Init(int* argc, char*** argv, int sync) {
  pthread_mutex_lock(&mutex_);
  if (init_) {
    pthread_mutex_unlock(&mutex_);
    return BRISBANE_ERR;
  }
  Utils::Logo(true);

  gethostname(brisbane_log_prefix_, 256);
  gethostname(host_, 256);
  if (argv && *argv) sprintf(app_, "%s", (*argv)[0]);
  else sprintf(app_, "%s", "app");

  timer_ = new Timer();
  timer_->Start(BRISBANE_TIMER_APP);

  timer_->Start(BRISBANE_TIMER_PLATFORM);
  const char* arch = getenv("BRISBANE_ARCH");
  if (!arch) arch = BRISBANE_ARCH;
  _info("Brisbane architectures[%s]", arch);
  const char* delim = " :;.,";
  char arch_str[32];
  memset(arch_str, 0, 32);
  strncpy(arch_str, arch, strlen(arch));
  for (char* a = strtok(arch_str, delim); a != NULL; a = strtok(NULL, delim)) {
    if (strcasecmp(a, "cuda") == 0) {
      if (!loaderCUDA_) InitCUDA();
    } else if (strcasecmp(a, "hip") == 0) {
      if (loaderHIP_ == NULL) InitHIP();
    } else if (strcasecmp(a, "opencl") == 0) {
      if (loaderOpenCL_ == NULL) InitOpenCL();
    } else if (strcasecmp(a, "openmp") == 0) {
      if (loaderOpenMP_ == NULL) InitOpenMP();
    } else _error("not support arch[%s]", a);
  }
  polyhedral_ = new Polyhedral();
  polyhedral_available_ = polyhedral_->Load() == BRISBANE_OK;
  if (polyhedral_available_)
    filter_task_split_ = new FilterTaskSplit(polyhedral_, this);

  brisbane_kernel null_brs_kernel;
  KernelCreate("brisbane_null", &null_brs_kernel);
  null_kernel_ = null_brs_kernel->class_obj;

  if (enable_profiler_) {
    profilers_[nprofilers_++] = new ProfilerDOT(this);
    profilers_[nprofilers_++] = new ProfilerGoogleCharts(this);
  }

  present_table_ = new PresentTable();

  scheduler_ = new Scheduler(this);
  scheduler_->Start();

  _info("nplatforms[%d] ndevs[%d] hub[%d] polyhedral[%d] profile[%d]", nplatforms_, ndevs_, scheduler_->hub_available(), polyhedral_available_, enable_profiler_);

  if (ndevs_) {
    device_default_ = 0;
    InitDevices(sync);
  } else {
    device_default_ = -1;
    __error("%s", "NO AVAILABLE DEVICES!");
  }

  timer_->Stop(BRISBANE_TIMER_PLATFORM);

  init_ = true;

  pthread_mutex_unlock(&mutex_);

  return BRISBANE_OK;
}

int Platform::Synchronize() {
  Task* task = new Task(this, BRISBANE_MARKER);
  for (int i = 0; i < ndevs_; i++)
    task->AddSubtask(new Task(this, BRISBANE_MARKER));
  scheduler_->Enqueue(task);
  task->Wait();
  return BRISBANE_OK;
}

int Platform::InitCUDA() {
  if (arch_available_ & brisbane_nvidia) {
    _trace("%s", "skipping CUDA architecture");
    return BRISBANE_ERR;
  }
  loaderCUDA_ = new LoaderCUDA();
  if (loaderCUDA_->Load() != BRISBANE_OK) {
    _trace("%s", "skipping CUDA architecture");
    return BRISBANE_ERR;
  }
  CUresult err = CUDA_SUCCESS;
  err = loaderCUDA_->cuInit(0);
  if (err != CUDA_SUCCESS) {
    _trace("skipping CUDA architecture CUDA_ERROR[%d]", err);
    return BRISBANE_ERR;
  }
  int ndevs = 0;
  err = loaderCUDA_->cuDeviceGetCount(&ndevs);
  _cuerror(err);
  _trace("CUDA platform[%d] ndevs[%d]", nplatforms_, ndevs);
  for (int i = 0; i < ndevs; i++) {
    CUdevice dev;
    err = loaderCUDA_->cuDeviceGet(&dev, i);
    _cuerror(err);
    devices_[ndevs_] = new DeviceCUDA(loaderCUDA_, dev, ndevs_, nplatforms_);
    arch_available_ |= devices_[ndevs_]->type();
    ndevs_++;
  }
  if (ndevs) {
    strcpy(platform_names_[nplatforms_], "CUDA");
    nplatforms_++;
  }
  return BRISBANE_OK;
}

int Platform::InitHIP() {
  if (arch_available_ & brisbane_amd) {
    _trace("%s", "skipping HIP architecture");
    return BRISBANE_ERR;
  }
  loaderHIP_ = new LoaderHIP();
  if (loaderHIP_->Load() != BRISBANE_OK) {
    _trace("%s", "skipping HIP architecture");
    return BRISBANE_ERR;
  }
  hipError_t err = hipSuccess;
  err = loaderHIP_->hipInit(0);
  _hiperror(err);
  int ndevs = 0;
  err = loaderHIP_->hipGetDeviceCount(&ndevs);
  _hiperror(err);
  _trace("HIP platform[%d] ndevs[%d]", nplatforms_, ndevs);
  for (int i = 0; i < ndevs; i++) {
    hipDevice_t dev;
    err = loaderHIP_->hipDeviceGet(&dev, i);
    _hiperror(err);
    devices_[ndevs_] = new DeviceHIP(loaderHIP_, dev, ndevs_, nplatforms_);
    arch_available_ |= devices_[ndevs_]->type();
    ndevs_++;
  }
  if (ndevs) {
    strcpy(platform_names_[nplatforms_], "HIP");
    nplatforms_++;
  }
  return BRISBANE_OK;
}

int Platform::InitOpenMP() {
  if (arch_available_ & brisbane_cpu) {
    _trace("%s", "skipping OpenMP architecture");
    return BRISBANE_ERR;
  }
  loaderOpenMP_ = new LoaderOpenMP();
  if (loaderOpenMP_->Load() != BRISBANE_OK) {
    _trace("%s", "skipping OpenMP architecture");
    return BRISBANE_ERR;
  }
  _trace("OpenMP platform[%d] ndevs[%d]", nplatforms_, 1);
  devices_[ndevs_] = new DeviceOpenMP(loaderOpenMP_, ndevs_, nplatforms_);
  arch_available_ |= devices_[ndevs_]->type();
  ndevs_++;
  strcpy(platform_names_[nplatforms_], "OpenMP");
  nplatforms_++;
  return BRISBANE_OK;
}

int Platform::InitOpenCL() {
  loaderOpenCL_ = new LoaderOpenCL();
  if (loaderOpenCL_->Load() != BRISBANE_OK) {
    _trace("%s", "skipping OpenCL architecture");
    return BRISBANE_ERR;
  }
  cl_platform_id cl_platforms[BRISBANE_MAX_NDEVS];
  cl_context cl_contexts[BRISBANE_MAX_NDEVS];
  cl_device_id cl_devices[BRISBANE_MAX_NDEVS];
  cl_int err;

  cl_uint nplatforms = BRISBANE_MAX_NDEVS;

  err = loaderOpenCL_->clGetPlatformIDs(nplatforms, cl_platforms, &nplatforms);
  _trace("OpenCL nplatforms[%u]", nplatforms);
  if (!nplatforms) return BRISBANE_OK;
  cl_uint ndevs = 0;
  char vendor[64];
  char platform_name[64];
  for (int i = 0; i < nplatforms; i++) {
    err = loaderOpenCL_->clGetPlatformInfo(cl_platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
    _clerror(err);
    err = loaderOpenCL_->clGetPlatformInfo(cl_platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    _clerror(err);

    if ((arch_available_ & brisbane_nvidia) && strstr(vendor, "NVIDIA") != NULL) {
      _trace("skipping platform[%d] [%s %s] ndevs[%u]", nplatforms_, vendor, platform_name, ndevs);
      continue;
    }
    if ((arch_available_ & brisbane_amd) && strstr(vendor, "Advanced Micro Devices") != NULL) {
      _trace("skipping platform[%d] [%s %s] ndevs[%u]", nplatforms_, vendor, platform_name, ndevs);
      continue;
    }
    err = loaderOpenCL_->clGetDeviceIDs(cl_platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &ndevs);
    if (!ndevs) {
      _trace("skipping platform[%d] [%s %s] ndevs[%u]", nplatforms_, vendor, platform_name, ndevs);
      continue;
    }
    err = loaderOpenCL_->clGetDeviceIDs(cl_platforms[i], CL_DEVICE_TYPE_ALL, ndevs, cl_devices, NULL);
    _clerror(err);
    cl_contexts[i] = loaderOpenCL_->clCreateContext(NULL, ndevs, cl_devices, NULL, NULL, &err);
    _clerror(err);
    if (err != CL_SUCCESS) {
      _trace("skipping platform[%d] [%s %s] ndevs[%u]", nplatforms_, vendor, platform_name, ndevs);
      continue;
    }
    for (cl_uint j = 0; j < ndevs; j++) {
      cl_device_type dev_type;
      err = loaderOpenCL_->clGetDeviceInfo(cl_devices[j], CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL);
      _clerror(err);
      if ((arch_available_ & brisbane_cpu) && (dev_type == CL_DEVICE_TYPE_CPU)) continue;
      devices_[ndevs_] = new DeviceOpenCL(loaderOpenCL_, cl_devices[j], cl_contexts[i], ndevs_, nplatforms_);
      arch_available_ |= devices_[ndevs_]->type();
      ndevs_++;
    }
    _trace("adding platform[%d] [%s %s] ndevs[%u]", nplatforms_, vendor, platform_name, ndevs);
    sprintf(platform_names_[nplatforms_], "OpenCL %s", vendor);
    nplatforms_++;
  }
  return BRISBANE_OK;
}

int Platform::InitDevices(bool sync) {
  Task** tasks = new Task*[ndevs_];
  for (int i = 0; i < ndevs_; i++) {
    tasks[i] = new Task(this);
    tasks[i]->set_system();
    Command* cmd = Command::CreateInit(tasks[i]);
    tasks[i]->AddCommand(cmd);
    scheduler_->worker(i)->Enqueue(tasks[i]);
  }
  if (sync) for (int i = 0; i < ndevs_; i++) tasks[i]->Wait();
  delete[] tasks;
  return BRISBANE_OK;
}

int Platform::PlatformCount(int* nplatforms) {
  if (nplatforms) *nplatforms = nplatforms_;
  return BRISBANE_OK;
}

int Platform::PlatformInfo(int platform, int param, void* value, size_t* size) {
  if (platform >= nplatforms_) return BRISBANE_ERR;
  switch (param) {
    case brisbane_name:
      if (size) *size = strlen(platform_names_[platform]);
      strcpy((char*) value, platform_names_[platform]);
      break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

int Platform::DeviceCount(int* ndevs) {
  if (ndevs) *ndevs = ndevs_;
  return BRISBANE_OK;
}

int Platform::DeviceInfo(int device, int param, void* value, size_t* size) {
  if (device >= ndevs_) return BRISBANE_ERR;
  Device* dev = devices_[device];
  switch (param) {
    case brisbane_platform  : if (size) *size = sizeof(int);            *((int*) value) = dev->platform();      break;
    case brisbane_vendor    : if (size) *size = strlen(dev->vendor());  strcpy((char*) value, dev->vendor());   break;
    case brisbane_name      : if (size) *size = strlen(dev->name());    strcpy((char*) value, dev->name());     break;
    case brisbane_type      : if (size) *size = sizeof(int);            *((int*) value) = dev->type();          break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

int Platform::DeviceSetDefault(int device) {
  device_default_ = device;
  return BRISBANE_OK;
}

int Platform::DeviceGetDefault(int* device) {
  *device = device_default_;
  return BRISBANE_OK;
}

int Platform::PolicyRegister(const char* lib, const char* name) {
  return scheduler_->policies()->Register(lib, name);
}

int Platform::KernelCreate(const char* name, brisbane_kernel* brs_kernel) {
  Kernel* kernel = new Kernel(name, this);
  if (brs_kernel) *brs_kernel = kernel->struct_obj();
  kernels_.insert(kernel);
  return BRISBANE_OK;
}

int Platform::KernelGet(const char* name, brisbane_kernel* brs_kernel) {
  for (std::set<Kernel*>::iterator I = kernels_.begin(), E = kernels_.end(); I != E; ++I) {
    Kernel* kernel = *I;
    if (strcmp(kernel->name(), name) == 0) {
      if (brs_kernel) *brs_kernel = kernel->struct_obj();
      return BRISBANE_OK;
    }
  }
  return KernelCreate(name, brs_kernel);
}

int Platform::KernelSetArg(brisbane_kernel brs_kernel, int idx, size_t size, void* value) {
  Kernel* kernel = brs_kernel->class_obj;
  kernel->SetArg(idx, size, value);
  return BRISBANE_OK;
}

int Platform::KernelSetMem(brisbane_kernel brs_kernel, int idx, brisbane_mem brs_mem, size_t off, int mode) {
  Kernel* kernel = brs_kernel->class_obj;
  Mem* mem = brs_mem->class_obj;
  kernel->SetMem(idx, mem, off, mode);
  return BRISBANE_OK;
}

int Platform::KernelSetMap(brisbane_kernel brs_kernel, int idx, void* host, int mode) {
  Kernel* kernel = brs_kernel->class_obj;
  size_t off = 0ULL;
  Mem* mem = present_table_->Get(host, &off);
  if (mem) kernel->SetMem(idx, mem, off, mode);
  else {
    _todo("clearing [%p]", host);
    MemMap(host, 8192);
    Mem* mem = present_table_->Get(host, &off);
    kernel->SetMem(idx, mem, off, mode);
  }
  return BRISBANE_OK;
}

int Platform::KernelRelease(brisbane_kernel brs_kernel) {
  Kernel* kernel = brs_kernel->class_obj;
  kernel->Release();
  return BRISBANE_OK;
}

int Platform::TaskCreate(const char* name, brisbane_task* brs_task) {
  Task* task = new Task(this, BRISBANE_TASK, name);
  *brs_task = task->struct_obj();
  return BRISBANE_OK;
}

int Platform::TaskDepend(brisbane_task brs_task, int ntasks, brisbane_task* brs_tasks) {
  Task* task = brs_task->class_obj;
  for (int i = 0; i < ntasks; i++) task->AddDepend(brs_tasks[i]->class_obj);
  return BRISBANE_OK;
}

int Platform::TaskKernel(brisbane_task brs_task, brisbane_kernel brs_kernel, int dim, size_t* off, size_t* ndr) {
  Task* task = brs_task->class_obj;
  Kernel* kernel = brs_kernel->class_obj;
  Command* cmd = Command::CreateKernel(task, kernel, dim, off, ndr, NULL);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::TaskKernel(brisbane_task brs_task, brisbane_kernel brs_kernel, int dim, size_t* off, size_t* ndr, size_t* lws) {
  Task* task = brs_task->class_obj;
  Kernel* kernel = brs_kernel->class_obj;
  Command* cmd = Command::CreateKernel(task, kernel, dim, off, ndr, lws);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::TaskH2D(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host) {
  Task* task = brs_task->class_obj;
  Mem* mem = brs_mem->class_obj;
  _debug("mem[%lu] off[%lu] size[%lu] host[%p]", mem->uid(), off, size, host);
  Command* cmd = Command::CreateH2D(task, mem, off, size, host);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::TaskD2H(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host) {
  Task* task = brs_task->class_obj;
  Mem* mem = brs_mem->class_obj;
  Command* cmd = Command::CreateD2H(task, mem, off, size, host);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::TaskH2DFull(brisbane_task brs_task, brisbane_mem brs_mem, void* host) {
  return TaskH2D(brs_task, brs_mem, 0ULL, brs_mem->class_obj->size(), host);
}

int Platform::TaskD2HFull(brisbane_task brs_task, brisbane_mem brs_mem, void* host) {
  return TaskD2H(brs_task, brs_mem, 0ULL, brs_mem->class_obj->size(), host);
}

int Platform::TaskMapTo(brisbane_task brs_task, void* host, size_t size) {
  Task* task = brs_task->class_obj;
  size_t off = 0ULL;
  Mem* mem = present_table_->Get(host, &off);
  Command* cmd = Command::CreateH2D(task, mem, off, size, host);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::TaskMapFrom(brisbane_task brs_task, void* host, size_t size) {
  Task* task = brs_task->class_obj;
  size_t off = 0ULL;
  Mem* mem = present_table_->Get(host, &off);
  Command* cmd = Command::CreateD2H(task, mem, off, size, host);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::TaskSubmit(brisbane_task brs_task, int brs_policy, const char* opt, int sync) {
  Task* task = brs_task->class_obj;
  task->set_brs_policy(brs_policy);
  task->set_opt(opt);
  task->set_sync(sync);
  FilterSubmitExecute(task);
  scheduler_->Enqueue(task);
  if (sync) task->Wait();
  return BRISBANE_OK;
}

int Platform::TaskWait(brisbane_task brs_task) {
  Task* task = brs_task->class_obj;
  task->Wait();
  return BRISBANE_OK;
}

int Platform::TaskWaitAll(int ntasks, brisbane_task* brs_tasks) {
  int iret = BRISBANE_OK;
  for (int i = 0; i < ntasks; i++) iret &= TaskWait(brs_tasks[i]);
  return iret;
}

int Platform::TaskAddSubtask(brisbane_task brs_task, brisbane_task brs_subtask) {
  Task* task = brs_task->class_obj;
  Task* subtask = brs_subtask->class_obj;
  task->AddSubtask(subtask);
  return BRISBANE_OK;
}

int Platform::TaskRelease(brisbane_task brs_task) {
  Task* task = brs_task->class_obj;
  task->Release();
  return BRISBANE_OK;
}

int Platform::TaskReleaseMem(brisbane_task brs_task, brisbane_mem brs_mem) {
  Task* task = brs_task->class_obj;
  Mem* mem = brs_mem->class_obj;
  Command* cmd = Command::CreateReleaseMem(task, mem);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::MemCreate(size_t size, brisbane_mem* brs_mem) {
  Mem* mem = new Mem(size, this);
  if (brs_mem) *brs_mem = mem->struct_obj();
  mems_.insert(mem);
  return BRISBANE_OK;
}

int Platform::MemMap(void* host, size_t size) {
  Mem* mem = new Mem(size, this);
  mem->SetMap(host, size);
  mems_.insert(mem);
  present_table_->Add(host, size, mem);
  return BRISBANE_OK;
}

int Platform::MemUnmap(void* host) {
  Mem* mem = present_table_->Remove(host);
  mem->Release();
  return BRISBANE_OK;
}

int Platform::MemReduce(brisbane_mem brs_mem, int mode, int type) {
  Mem* mem = brs_mem->class_obj;
  mem->Reduce(mode, type);
  return BRISBANE_OK;
}

int Platform::MemRelease(brisbane_mem brs_mem) {
  Mem* mem = brs_mem->class_obj;
  mem->Release();
  return BRISBANE_OK;
}

int Platform::FilterSubmitExecute(Task* task) {
  if (!polyhedral_available_) return BRISBANE_OK;
  if (!task->cmd_kernel()) return BRISBANE_OK;
  if (task->brs_policy() & brisbane_all) {
    if (filter_task_split_->Execute(task) != BRISBANE_OK) {
      _trace("poly is not available kernel[%s] task[%lu]", task->cmd_kernel()->kernel()->name(), task->uid());
      return BRISBANE_ERR;
    }
    _trace("poly is available kernel[%s] task[%lu]", task->cmd_kernel()->kernel()->name(), task->uid());
  }
  return BRISBANE_OK;
}

int Platform::TimerNow(double* time) {
  *time = timer_->Now();
  return BRISBANE_OK;
}

int Platform::ShowKernelHistory() {
  double t_ker = 0.0;
  double t_h2d = 0.0;
  double t_d2h = 0.0;
  for (std::set<Kernel*>::iterator I = kernels_.begin(), E = kernels_.end(); I != E; ++I) {
    Kernel* kernel = *I;
    History* history = kernel->history();
    _info("kernel[%s] k[%lf][%lu] h2d[%lf][%lu] d2h[%lf][%lu]", kernel->name(), history->t_kernel(), history->c_kernel(), history->t_h2d(), history->c_h2d(), history->t_d2h(), history->c_d2h());
    t_ker += history->t_kernel();
    t_h2d += history->t_h2d();
    t_d2h += history->t_d2h();
  }
  _info("total kernel[%lf] h2d[%lf] d2h[%lf]", t_ker, t_h2d, t_d2h);
  return BRISBANE_OK;
}

Platform* Platform::singleton_ = NULL;
std::once_flag Platform::flag_singleton_;
std::once_flag Platform::flag_finalize_;

Platform* Platform::GetPlatform() {
//  if (singleton_ == NULL) singleton_ = new Platform();
  std::call_once(flag_singleton_, []() { singleton_ = new Platform(); });
  return singleton_;
}

int Platform::Finalize() {
  pthread_mutex_lock(&mutex_);
  if (finalize_) {
    pthread_mutex_unlock(&mutex_);
    return BRISBANE_ERR;
  }
  Synchronize();
  ShowKernelHistory();
  time_app_ = timer()->Stop(BRISBANE_TIMER_APP);
  time_init_ = timer()->Total(BRISBANE_TIMER_PLATFORM);
  _info("total execution time:[%lf] sec. initialize:[%lf] sec. t-i:[%lf] sec", time_app_, time_init_, time_app_ - time_init_);
  finalize_ = true;
  pthread_mutex_unlock(&mutex_);
  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */

