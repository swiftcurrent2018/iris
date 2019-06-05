#include "Task.h"
#include "Debug.h"
#include "Command.h"
#include "Device.h"
#include "Kernel.h"
#include "Mem.h"

namespace brisbane {
namespace rt {

Task::Task(Platform* platform) {
    ncmds_ = 0;
    cmd_kernel_ = NULL;
    platform_ = platform;
    dev_ = NULL;
    status_ = BRISBANE_NONE;

    pthread_mutex_init(&complete_mutex_, NULL);
    pthread_cond_init(&complete_cond_, NULL);
}

Task::~Task() {
    pthread_mutex_destroy(&complete_mutex_);
    pthread_cond_destroy(&complete_cond_);
}

void Task::AddCommand(Command* cmd) {
    cmds_[ncmds_++] = cmd;
    if (cmd->type() == BRISBANE_CMD_KERNEL) {
        if (cmd_kernel_) _error("kernel[%s] is already set", cmd->kernel()->name());
        cmd_kernel_ = cmd;
    }
}

void Task::Execute() {
    dev_->Execute(this);
}

void Task::Complete() {
    pthread_mutex_lock(&complete_mutex_);
    status_ = BRISBANE_COMPLETE;
    pthread_cond_broadcast(&complete_cond_);
    pthread_mutex_unlock(&complete_mutex_);
}

void Task::Wait() {
    pthread_mutex_lock(&complete_mutex_);
    if (status_ != BRISBANE_COMPLETE)
        pthread_cond_wait(&complete_cond_, &complete_mutex_);
    pthread_mutex_unlock(&complete_mutex_);
}

void Task::AddSubtask(Task* subtask) {
    subtasks_.push_back(subtask);
}

bool Task::HasSubtasks() {
    return !subtasks_.empty();
}

} /* namespace rt */
} /* namespace brisbane */
