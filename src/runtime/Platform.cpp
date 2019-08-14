#include "Platform.h"
#include "Utils.h"
#include "Command.h"
#include "Device.h"
#include "FilterTaskSplit.h"
#include "History.h"
#include "Kernel.h"
#include "Mem.h"
#include "Polyhedral.h"
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
  nplatforms_ = BRISBANE_MAX_NDEVS;
  ndevs_ = 0;

  scheduler_ = NULL;
  polyhedral_ = NULL;
  filter_task_split_ = NULL;
  timer_ = NULL;
  null_kernel_ = NULL;
  enable_profiler_ = getenv("BRISBANE_PROFILE");
  nprofilers_ = 0;
  time_app_ = 0.0;
  time_init_ = 0.0;
}

Platform::~Platform() {
  if (!init_) return;

  if (scheduler_) delete scheduler_;
  if (polyhedral_) delete polyhedral_;
  if (filter_task_split_) delete filter_task_split_;
  if (timer_) delete timer_;
  if (null_kernel_) delete null_kernel_;
  if (enable_profiler_)
    for (int i = 0; i < nprofilers_; i++) delete profilers_[i];
}

int Platform::Init(int* argc, char*** argv, int sync) {
  if (init_) return BRISBANE_ERR;
  Utils::Logo(true);

  gethostname(brisbane_log_prefix_, 256);
  gethostname(host_, 256);
  if (argv && *argv) sprintf(app_, "%s", (*argv)[0]);
  else sprintf(app_, "%s", "app");

  timer_ = new Timer();
  timer_->Start(BRISBANE_TIMER_APP);

  timer_->Start(BRISBANE_TIMER_INIT);
  InitCLPlatforms();
  polyhedral_ = new Polyhedral(this);
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

  _info("available: polyhedral[%d] profile[%d]", polyhedral_available_, enable_profiler_);

  scheduler_ = new Scheduler(this);
  scheduler_->Start();

  BuildPrograms(sync);

  init_ = true;

  timer_->Stop(BRISBANE_TIMER_INIT);

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

int Platform::InitCLPlatforms() {
  bool enable = true;

  clerr_ = clGetPlatformIDs((cl_uint) nplatforms_, cl_platforms_, (cl_uint*) &nplatforms_);
  _info("nplatforms[%u]", nplatforms_);
  cl_uint num_devices;
  char platform_vendor[64];
  char platform_name[64];
  for (int i = 0; i < nplatforms_; i++) {
    clerr_ = clGetPlatformInfo(cl_platforms_[i], CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, NULL);
    clerr_ = clGetPlatformInfo(cl_platforms_[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    clerr_ = clGetDeviceIDs(cl_platforms_[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    clerr_ = clGetDeviceIDs(cl_platforms_[i], CL_DEVICE_TYPE_ALL, num_devices, cl_devices_ + ndevs_, NULL);
    cl_contexts_[i] = clCreateContext(NULL, num_devices, cl_devices_ + ndevs_, NULL, NULL, &clerr_);
    _clerror(clerr_);
    for (cl_uint j = 0; j < num_devices; j++) {
      devices_[ndevs_] = new Device(cl_devices_[ndevs_], cl_contexts_[i], ndevs_, i);
      //enable &= devices_[ndevs_]->enable();
      ndevs_++;
    }
  }
  if (ndevs_) device_default_ = devices_[0]->type();
  if (!enable) exit(-1);
  return BRISBANE_OK;
}

int Platform::BuildPrograms(bool sync) {
  Task** tasks = new Task*[ndevs_];
  for (int i = 0; i < ndevs_; i++) {
    tasks[i] = new Task(this);
    tasks[i]->set_system();
    Command* cmd = Command::CreateBuild(tasks[i]);
    tasks[i]->AddCommand(cmd);
    scheduler_->worker(i)->Enqueue(tasks[i]);
  }
  if (sync) {
    for (int i = 0; i < ndevs_; i++) {
      tasks[i]->Wait();
      delete tasks[i];
    }
  }
  delete[] tasks;
  return BRISBANE_OK;
}

int Platform::InfoNumPlatforms(int* nplatforms) {
  *nplatforms = nplatforms_;
  return BRISBANE_OK;
}

int Platform::InfoNumDevices(int* ndevs) {
  *ndevs = ndevs_;
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

int Platform::KernelCreate(const char* name, brisbane_kernel* brs_kernel) {
  for (std::set<Kernel*>::iterator I = kernels_.begin(), E = kernels_.end(); I != E; ++I) {
    Kernel* kernel = *I;
    if (strcmp(kernel->name(), name) == 0) {
      if (brs_kernel) *brs_kernel = kernel->struct_obj();
      return BRISBANE_OK;
    }
  }
  Kernel* kernel = new Kernel(name, this);
  if (brs_kernel) *brs_kernel = kernel->struct_obj();
  kernels_.insert(kernel);
  return BRISBANE_OK;
}

int Platform::KernelSetArg(brisbane_kernel brs_kernel, int idx, size_t arg_size, void* arg_value) {
  Kernel* kernel = brs_kernel->class_obj;
  kernel->SetArg(idx, arg_size, arg_value);
  return BRISBANE_OK;
}

int Platform::KernelSetMem(brisbane_kernel brs_kernel, int idx, brisbane_mem brs_mem, int mode) {
  Kernel* kernel = brs_kernel->class_obj;
  Mem* mem = brs_mem->class_obj;
  kernel->SetMem(idx, mem, mode);
  return BRISBANE_OK;
}

int Platform::KernelRelease(brisbane_kernel brs_kernel) {
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
  Command* cmd = Command::CreateKernel(task, kernel, dim, off, ndr);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::TaskH2D(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host) {
  Task* task = brs_task->class_obj;
  Mem* mem = brs_mem->class_obj;
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

int Platform::TaskPresent(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host) {
  Task* task = brs_task->class_obj;
  Mem* mem = brs_mem->class_obj;
  Command* cmd = Command::CreatePresent(task, mem, off, size, host);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::TaskSubmit(brisbane_task brs_task, int brs_device, char* opt, int sync) {
  Task* task = brs_task->class_obj;
  task->set_brs_device(brs_device);
  FilterSubmitExecute(task);
  scheduler_->Enqueue(task, sync);
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
  delete task;
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
  *brs_mem = mem->struct_obj();
  mems_.insert(mem);
  return BRISBANE_OK;
}

int Platform::MemReduce(brisbane_mem brs_mem, int mode, int type) {
  Mem* mem = brs_mem->class_obj;
  mem->Reduce(mode, type);
  return BRISBANE_OK;
}

int Platform::MemRelease(brisbane_mem brs_mem) {
  Mem* mem = brs_mem->class_obj;
  delete mem;
  return BRISBANE_OK;
}

int Platform::FilterSubmitExecute(Task* task) {
  if (!polyhedral_available_) return BRISBANE_OK;
  if (!task->cmd_kernel()) return BRISBANE_OK;
  if (task->brs_device() & brisbane_all) {
    if (filter_task_split_->Execute(task) != BRISBANE_OK) {
      _debug("poly is not available kernel[%s] task[%lu]", task->cmd_kernel()->kernel()->name(), task->uid());
      return BRISBANE_ERR;
    }
    _debug("poly is available kernel[%s] task[%lu]", task->cmd_kernel()->kernel()->name(), task->uid());
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

Platform* Platform::GetPlatform() {
  if (singleton_ == NULL) singleton_ = new Platform();
  return singleton_;
}

int Platform::Finalize() {
  singleton_->Synchronize();
  singleton_->ShowKernelHistory();
  singleton_->time_app_ = singleton_->timer()->Stop(BRISBANE_TIMER_APP);
  singleton_->time_init_ = singleton_->timer()->Total(BRISBANE_TIMER_INIT);
  double time_app = singleton_->time_app();
  double time_init = singleton_->time_init();
  if (singleton_ == NULL) return BRISBANE_ERR;
  if (singleton_) delete singleton_;
  singleton_ = NULL;
  _info("total execution time:[%lf] sec. initialize:[%lf] sec. t-i:[%lf] sec", time_app, time_init, time_app - time_init);
  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */
