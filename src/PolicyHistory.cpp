#include "PolicyHistory.h"
#include "Debug.h"
#include "Command.h"
#include "History.h"
#include "Policies.h"
#include "Kernel.h"
#include "Task.h"

namespace brisbane {
namespace rt {

PolicyHistory::PolicyHistory(Scheduler* scheduler, Policies* policies) {
    SetScheduler(scheduler);
    policies_ = policies;
}

PolicyHistory::~PolicyHistory() {
}

void PolicyHistory::GetDevices(Task* task, Device** devs, int* ndevs) {
    Command* cmd = task->cmd_kernel();
    if (!cmd) return policies_->GetPolicy(brisbane_device_default)->GetDevices(task, devs, ndevs);
    History* history = cmd->kernel()->history();
    devs[0] = history->OptimalDevice(task);
    *ndevs = 1;
}

} /* namespace rt */
} /* namespace brisbane */
