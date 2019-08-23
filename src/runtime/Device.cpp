#include "Device.h"
#include "Command.h"
#include "History.h"
#include "Kernel.h"
#include "Mem.h"
#include "Reduction.h"
#include "Task.h"
#include "Timer.h"
#include "Utils.h"

namespace brisbane {
namespace rt {

Device::Device(int devno, int platform) {
  devno_ = devno;
  platform_ = platform;
  busy_ = false;
  enable_ = false;
  timer_ = new Timer();
}

Device::~Device() {
  delete timer_;
}

void Device::Execute(Task* task) {
  busy_ = true;
  for (int i = 0; i < task->ncmds(); i++) {
    Command* cmd = task->cmd(i);
    switch (cmd->type()) {
      case BRISBANE_CMD_INIT:         ExecuteInit(cmd);       break;
      case BRISBANE_CMD_KERNEL:       ExecuteKernel(cmd);     break;
      case BRISBANE_CMD_H2D:          ExecuteH2D(cmd);        break;
      case BRISBANE_CMD_H2DNP:        ExecuteH2DNP(cmd);      break;
      case BRISBANE_CMD_D2H:          ExecuteD2H(cmd);        break;
      case BRISBANE_CMD_RELEASE_MEM:  ExecuteReleaseMem(cmd); break;
      default: _error("cmd type[0x%x]", cmd->type());
    }
  }
  _info("task[%lu][%s] complete dev[%d][%s] time[%lf]", task->uid(), task->name(), devno(), name(), task->time());
  busy_ = false;
}

void Device::ExecuteInit(Command* cmd) {
  timer_->Start(BRISBANE_TIMER_INIT);
  int iret = Init();
  if (iret != BRISBANE_OK) _error("iret[%d]", iret);
  double time = timer_->Stop(BRISBANE_TIMER_INIT);
  cmd->SetTime(time);
  enable_ = true;
}

void Device::ExecuteKernel(Command* cmd) {
  timer_->Start(BRISBANE_TIMER_KERNEL);

  Kernel* kernel = cmd->kernel();
  int dim = cmd->dim();
  size_t* off = cmd->off();
  size_t* gws = cmd->ndr();
  size_t gws0 = gws[0];
  size_t* lws = NULL;
  bool reduction = false;
  brisbane_poly_mem* polymems = cmd->polymems();
  int npolymems = cmd->npolymems();
  int max_idx = 0;
  int mem_idx = 0;
  KernelLaunchInit(kernel);
//  cl_kernel clkernel = kernel->clkernel(devno_, clprog_);
  std::map<int, KernelArg*>* args = cmd->kernel_args();
  for (std::map<int, KernelArg*>::iterator I = args->begin(), E = args->end(); I != E; ++I) {
    int idx = I->first;
    if (idx > max_idx) max_idx = idx;
    KernelArg* arg = I->second;
    Mem* mem = arg->mem;
    if (mem) {
      if (arg->mode & brisbane_w) {
        if (npolymems) {
          brisbane_poly_mem* pm = polymems + mem_idx;
          mem->SetOwner(pm->typesz * pm->w0, pm->typesz * (pm->w1 - pm->w0 + 1), this);
        } else mem->SetOwner(this);
      }
      if (mem->mode() & brisbane_reduction) {
        lws = (size_t*) alloca(3 * sizeof(size_t));
        lws[0] = 1;
        lws[1] = 1;
        lws[2] = 1;
        while (max_compute_units_ * lws[0] < gws[0]) lws[0] <<= 1;
        while (max_work_item_sizes_[0] / 4 < lws[0]) lws[0] >>= 1;
        size_t expansion = (gws[0] + lws[0] - 1) / lws[0];
        gws[0] = lws[0] * expansion;
        mem->Expand(expansion);
        KernelSetArg(kernel, idx + 1, lws[0] * mem->type_size(), NULL);
        /*
        err_ = clSetKernelArg(clkernel, (cl_uint) idx + 1, lws[0] * mem->type_size(), NULL);
        _clerror(err_);
        */
        reduction = true;
        if (idx + 1 > max_idx) max_idx = idx + 1;
      }
      KernelSetMem(kernel, idx, mem);
      /*
      cl_mem clmem = mem->clmem(platform_, clctx_);
      err_ = clSetKernelArg(clkernel, (cl_uint) idx, sizeof(clmem), (const void*) &clmem);
      _clerror(err_);
      */
      mem_idx++;
    } else {
      KernelSetArg(kernel, idx, arg->size, arg->value);
      /*
      err_ = clSetKernelArg(clkernel, (cl_uint) idx, arg->size, (const void*) arg->value);
      _clerror(err_);
      */
    }
  }
  if (reduction) {
    _trace("max_idx+1[%d] gws[%lu]", max_idx + 1, gws0);
    KernelSetArg(kernel, max_idx + 1, sizeof(size_t), &gws0);
    /*
    err_ = clSetKernelArg(clkernel, (cl_uint) max_idx + 1, sizeof(size_t), &gws0);
    _clerror(err_);
    */
  }

  KernelLaunch(kernel, dim, off, gws, lws);

  //_trace("devno[%d][%s] kernel[%s] dim[%d] off[%lu,%lu,%lu] gws[%lu,%lu,%lu] lws[%lu,%lu,%lu]", devno_, name_, kernel->name(), dim, off[0], off[1], off[2], gws[0], gws[1], gws[2], lws ? lws[0] : 0, lws ? lws[1] : 0, lws ? lws[2] : 0);
  /*
  if (lws && (lws[0] > gws[0] || lws[1] > gws[1] || lws[2] > gws[2])) _error("gws[%lu,%lu,%lu] and lws[%lu,%lu,%lu]", gws[0], gws[1], gws[2], lws[0], lws[1], lws[2]);
  if (type_ == brisbane_fpga) {
    if (off[0] != 0 || off[1] != 0 || off[2] != 0)
      _todo("%s", "global_work_offset shoule be set to not NULL. Upgrade Intel FPGA SDK for OpenCL Pro Edition Version 19.1");
    err_ = clEnqueueNDRangeKernel(clcmdq_, clkernel, (cl_uint) dim, NULL, (const size_t*) gws, (const size_t*) lws, 0, NULL, NULL);
  } else {
    err_ = clEnqueueNDRangeKernel(clcmdq_, clkernel, (cl_uint) dim, (const size_t*) off, (const size_t*) gws, (const size_t*) lws, 0, NULL, NULL);
  }
  _clerror(err_);
  err_ = clFinish(clcmdq_);
  _clerror(err_);
  */

  double time = timer_->Stop(BRISBANE_TIMER_KERNEL);
  cmd->SetTime(time);
  cmd->kernel()->history()->AddKernel(cmd, this, time);
}

void Device::ExecuteH2D(Command* cmd) {
  Mem* mem = cmd->mem();
  size_t off = cmd->off(0);
  size_t size = cmd->size();
  bool exclusive = cmd->exclusive();
  void* host = cmd->host();
  if (exclusive) mem->SetOwner(off, size, this);
  else mem->AddOwner(off, size, this);
  timer_->Start(BRISBANE_TIMER_H2D);
  int iret = H2D(mem, off, size, host);
  if (iret != BRISBANE_OK) _error("iret[%d]", iret);
  double time = timer_->Stop(BRISBANE_TIMER_H2D);
  cmd->SetTime(time);
  Command* cmd_kernel = cmd->task()->cmd_kernel();
  if (cmd_kernel) cmd_kernel->kernel()->history()->AddH2D(cmd, this, time);
  else Platform::GetPlatform()->null_kernel()->history()->AddH2D(cmd, this, time);
}

void Device::ExecuteH2DNP(Command* cmd) {
  Mem* mem = cmd->mem();
  size_t off = cmd->off(0);
  size_t size = cmd->size();
  if (mem->IsOwner(off, size, this)) return;
  return ExecuteH2D(cmd);
}

void Device::ExecuteD2H(Command* cmd) {
  Mem* mem = cmd->mem();
  size_t off = cmd->off(0);
  size_t size = cmd->size();
  void* host = cmd->host();
  int mode = mem->mode();
  int expansion = mem->expansion();
  timer_->Start(BRISBANE_TIMER_D2H);
  int iret = BRISBANE_OK;
  if (mode & brisbane_reduction) {
    iret = D2H(mem, off, mem->size() * expansion, mem->host_inter());
    Reduction::GetInstance()->Reduce(mem, host, size);
  } else iret = D2H(mem, off, size, host);
  if (iret != BRISBANE_OK) _error("iret[%d]", iret);
  double time = timer_->Stop(BRISBANE_TIMER_D2H);
  cmd->SetTime(time);
  Command* cmd_kernel = cmd->task()->cmd_kernel();
  if (cmd_kernel) cmd_kernel->kernel()->history()->AddD2H(cmd, this, time);
  else Platform::GetPlatform()->null_kernel()->history()->AddD2H(cmd, this, time);
}

void Device::ExecuteReleaseMem(Command* cmd) {
  Mem* mem = cmd->mem();
  mem->Release(); 
}

} /* namespace rt */
} /* namespace brisbane */
