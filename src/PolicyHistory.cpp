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

Device* PolicyHistory::GetDevice(Task* task) {
    Command* cmd = task->cmd_kernel();
    if (!cmd) return policies_->GetPolicy(brisbane_device_default)->GetDevice(task);
    History* history = cmd->kernel()->history();
    return history->OptimalDevice(task);
}

} /* namespace rt */
} /* namespace brisbane */
