#include "Consistency.h"
#include "Debug.h"
#include "Device.h"
#include "Command.h"
#include "Kernel.h"
#include "Mem.h"
#include "Scheduler.h"
#include "Task.h"

namespace brisbane {
namespace rt {

Consistency::Consistency(Scheduler* scheduler) {
  scheduler_ = scheduler;
}

Consistency::~Consistency() {
}

void Consistency::Resolve(Task* task) {
  for (int i = 0; i < task->ncmds(); i++) {
    Command* cmd = task->cmd(i);
    if (cmd->type() != BRISBANE_CMD_KERNEL) continue;
    //TODO: handle others cmds
    Resolve(task, cmd);
  }
}

void Consistency::Resolve(Task* task, Command* cmd) {
//  if (task->parent()) return;
  Device* dev = task->dev();
  Kernel* kernel = cmd->kernel();
  brisbane_poly_mem* polymems = cmd->polymems();
  int npolymems = cmd->npolymems();
  std::map<int, KernelArg*>* args = kernel->args();
  int mem_idx = 0;
  for (std::map<int, KernelArg*>::iterator I = args->begin(), E = args->end(); I != E; ++I) {
    KernelArg* arg = I->second;
    Mem* mem = I->second->mem;
    if (!mem) continue;

    if (npolymems) ResolveWithPolymem(task, cmd, mem, arg, polymems + mem_idx);
    else ResolveWithoutPolymem(task, cmd, mem);

    mem_idx++;
  }
}

void Consistency::ResolveWithPolymem(Task* task, Command* cmd, Mem* mem, KernelArg* arg, brisbane_poly_mem* polymem) {
  Device* dev = task->dev();
  Kernel* kernel = cmd->kernel();
  size_t off = 0UL;
  size_t size = 0UL;
  if (arg->mode == brisbane_r) {
    off = polymem->typesz * polymem->r0;
    size = polymem->typesz * (polymem->r1 - polymem->r0 + 1);
  } else if (arg->mode == brisbane_w) {
    off = polymem->typesz * polymem->w0;
    size = polymem->typesz * (polymem->w1 - polymem->w0 + 1);
  } else if (arg->mode == brisbane_rw) {
    off = polymem->r0 < polymem->w0 ? polymem->r0 : polymem->w0;
    size = polymem->typesz * (polymem->r1 > polymem->w1 ? polymem->r1 - off + 1 : polymem->w1 - off + 1);
    off *= polymem->typesz;
  } else _error("not supprt mode[0x%x]", arg->mode);

  Device* owner = mem->Owner(off, size);
  if (!owner || mem->IsOwner(off, size, dev)) return;

  Task* task_d2h = new Task(scheduler_->platform());
  task_d2h->set_system();
  Command* d2h = Command::CreateD2H(task, mem, off, size, (char*) mem->host_inter() + off);
  task_d2h->AddCommand(d2h);
  scheduler_->SubmitTaskDirect(task_d2h, owner);
  task_d2h->Wait();

  Command* h2d = arg->mode == brisbane_r ?
    Command::CreateH2DNP(task, mem, off, size, (char*) mem->host_inter() + off) :
    Command::CreateH2D(task, mem, off, size, (char*) mem->host_inter() + off);
  dev->ExecuteH2D(h2d);

  _trace("kernel[%s] memcpy[%lu] [%s] -> [%s]", kernel->name(), mem->uid(), owner->name(), dev->name());

  task_d2h->Release();
  Command::Release(h2d);
}

void Consistency::ResolveWithoutPolymem(Task* task, Command* cmd, Mem* mem) {
  Device* dev = task->dev();
  Kernel* kernel = cmd->kernel();
  Device* owner = mem->Owner();
  if (!owner || mem->IsOwner(0, mem->size(), dev)) return;

  Task* task_d2h = new Task(scheduler_->platform());
  task_d2h->set_system();
  Command* d2h = Command::CreateD2H(task_d2h, mem, 0, mem->size(), mem->host_inter());
  task_d2h->AddCommand(d2h);
  scheduler_->SubmitTaskDirect(task_d2h, owner);
  task_d2h->Wait();

  Command* h2d = Command::CreateH2D(task, mem, 0, mem->size(), mem->host_inter());
  dev->ExecuteH2D(h2d);

  _trace("kernel[%s] memcpy[%lu] [%s] -> [%s]", kernel->name(), mem->uid(), owner->name(), dev->name());

  task_d2h->Release();
  Command::Release(h2d);
}

} /* namespace rt */
} /* namespace brisbane */

