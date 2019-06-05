#include "Executor.h"
#include "Debug.h"
#include "Device.h"
#include "Command.h"
#include "Kernel.h"
#include "Mem.h"
#include "Task.h"

namespace brisbane {
namespace rt {

Executor::Executor() {
}

Executor::~Executor() {
}

void Executor::Execute(Task* task) {
    for (int i = 0; i < task->ncmds(); i++) {
        Command* cmd = task->cmd(i);
        if (cmd->type() != BRISBANE_CMD_KERNEL) continue;
        ResolveDependency(cmd, task);
    }
    task->Execute();
}

void Executor::ResolveDependency(Command* cmd, Task* task) {
    Device* dev = task->dev();
    Kernel* kernel = cmd->kernel();
    std::map<int, KernelArg*>* args = kernel->args();
    for (std::map<int, KernelArg*>::iterator it = args->begin(); it != args->end(); ++it) {
        KernelArg* arg = it->second;
        Mem* mem = it->second->mem;
        if (!mem || mem->IsOwner(dev)) continue;

        Device* owner = mem->owner();
        Command* d2h = Command::CreateD2H(mem, 0, mem->size(), mem->host_inter());
        owner->ExecuteD2H(d2h);

        Command* h2d = Command::CreateH2D(mem, 0, mem->size(), mem->host_inter());
        dev->ExecuteH2D(h2d);

        _trace("kernel[%s] memcpy[%lu] [%s] -> [%s]", kernel->name(), mem->uid(), owner->name(), dev->name());

        Command::Release(d2h);
        Command::Release(h2d);
    }
}

} /* namespace rt */
} /* namespace brisbane */
