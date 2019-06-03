#include "Task.h"
#include "Debug.h"
#include "Command.h"
#include "Device.h"
#include "Kernel.h"
#include "Mem.h"

namespace brisbane {
namespace rt {

Task::Task(Task* parent) {
    parent_ = parent;
    num_cmds_ = 0;
    cmd_kernel_ = NULL;
    platform_ = Platform::GetPlatform();
    if (parent_) parent_->AddSubtask(this); 
}

Task::~Task() {
}

void Task::AddCommand(Command* cmd) {
    cmds_[num_cmds_++] = cmd;
    if (cmd->type() == BRISBANE_CMD_KERNEL) {
        if (cmd_kernel_) _error("kernel[%s] is already set", cmd->kernel()->name());
        cmd_kernel_ = cmd;
    }
}

void Task::Submit(int brs_device) {
    int target_brs_device = brs_device;
    dev_ = platform_->AvailableDevice(this, target_brs_device);
    if (dev_ == NULL) dev_ = platform_->device(0);
    platform_->ExecuteTask(this);
}

void Task::Execute() {
    dev_->Execute(this);
}

void Task::Wait() {
    dev_->Wait();
}

void Task::AddSubtask(Task* subtask) {
    subtasks_.push_back(subtask);
}

} /* namespace rt */
} /* namespace brisbane */
