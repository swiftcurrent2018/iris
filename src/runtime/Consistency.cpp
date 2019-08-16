#include "Consistency.h"
#include "Debug.h"
#include "Device.h"
#include "Command.h"
#include "Kernel.h"
#include "Mem.h"
#include "Task.h"

namespace brisbane {
namespace rt {

Consistency::Consistency() {
}

Consistency::~Consistency() {
}

void Consistency::Resolve(Task* task) {
  for (int i = 0; i < task->ncmds(); i++) {
    Command* cmd = task->cmd(i);
    if (cmd->type() != BRISBANE_CMD_KERNEL) continue;
    Resolve(task, cmd);
  }
}

void Consistency::Resolve(Task* task, Command* cmd) {
  if (task->parent()) return;
  Device* dev = task->dev();
  Kernel* kernel = cmd->kernel();
  std::map<int, KernelArg*>* args = kernel->args();
  for (std::map<int, KernelArg*>::iterator I = args->begin(), E = args->end(); I != E; ++I) {
    KernelArg* arg = I->second;
    Mem* mem = I->second->mem;
    if (!mem || mem->EmptyOwner() || mem->IsOwner(dev)) continue;

    Device* owner = mem->owner();
    _debug("mem[%lu] owner[%p] dev[%d]", mem->uid(), owner, dev->devno());
    Command* d2h = Command::CreateD2H(task, mem, 0, mem->size(), mem->host_inter());
    owner->ExecuteD2H(d2h);

    Command* h2d = Command::CreateH2D(task, mem, 0, mem->size(), mem->host_inter());
    dev->ExecuteH2D(h2d);

    _trace("kernel[%s] memcpy[%lu] [%s] -> [%s]", kernel->name(), mem->uid(), owner->name(), dev->name());

    Command::Release(d2h);
    Command::Release(h2d);
  }
}

} /* namespace rt */
} /* namespace brisbane */
