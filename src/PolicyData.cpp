#include "PolicyData.h"
#include "Debug.h"
#include "Command.h"
#include "Device.h"
#include "Kernel.h"
#include "Mem.h"
#include "Task.h"

namespace brisbane {
namespace rt {

PolicyData::PolicyData(Scheduler* scheduler) {
    SetScheduler(scheduler);
}

PolicyData::~PolicyData() {
}

Device* PolicyData::GetDevice(Task* task) {
    size_t total_size[BRISBANE_MAX_NDEVS];
    for (int i = 0; i < ndevs_; i++) total_size[i] = 0UL;
    for (int i = 0; i < task->ncmds(); i++) {
        Command* cmd = task->cmd(i);
        if (cmd->type() == BRISBANE_CMD_KERNEL) {
            Kernel* kernel = cmd->kernel();
            std::map<int, KernelArg*>* args = kernel->args();
            for (std::map<int, KernelArg*>::iterator it = args->begin(); it != args->end(); ++it) {
                Mem* mem = it->second->mem;
                if (!mem || !mem->owner()) continue;
                total_size[mem->owner()->dev_no()] += mem->size();
            }
        } else if (cmd->type() == BRISBANE_CMD_H2D || cmd->type() == BRISBANE_CMD_D2H) {
            Mem* mem = cmd->mem();
            if (!mem || !mem->owner()) continue;
            total_size[mem->owner()->dev_no()] += mem->size();
        }
    }
    int target_dev = 0;
    size_t max_size = 0UL;
    for (int i = 0; i < ndevs_; i++) {
        if (total_size[i] > max_size) {
            max_size = total_size[i];
            target_dev = i;
        }
    }
    return devices_[target_dev];
}

} /* namespace rt */
} /* namespace brisbane */
