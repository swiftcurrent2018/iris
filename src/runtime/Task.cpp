#include "Task.h"
#include "Debug.h"
#include "Command.h"
#include "Device.h"
#include "Kernel.h"
#include "Mem.h"
#include "Scheduler.h"

namespace brisbane {
namespace rt {

Task::Task(Platform* platform, int type, const char* name) {
  type_ = type;
  ncmds_ = 0;
  cmd_kernel_ = NULL;
  cmd_last_ = NULL;
  platform_ = platform;
  scheduler_ = platform->scheduler();
  dev_ = NULL;
  parent_ = NULL;
  subtasks_complete_ = 0;
  ndepends_ = 0;
  sync_ = false;
  time_ = 0.0;
  time_start_ = 0.0;
  time_end_ = 0.0;
  system_ = false;
  given_name_ = name != NULL;
  if (name) strcpy(name_, name);
  else sprintf(name_, "task%ld", uid());
  status_ = BRISBANE_NONE;

  pthread_mutex_init(&mutex_executable_, NULL);
  pthread_mutex_init(&mutex_complete_, NULL);
  pthread_mutex_init(&mutex_subtasks_, NULL);
  pthread_cond_init(&complete_cond_, NULL);
}

Task::~Task() {
  for (int i = 0; i < ncmds_; i++) delete cmds_[i];
  pthread_mutex_destroy(&mutex_executable_);
  pthread_mutex_destroy(&mutex_complete_);
  pthread_mutex_destroy(&mutex_subtasks_);
  pthread_cond_destroy(&complete_cond_);
}

double Task::TimeInc(double t) {
  time_ += t;
  return time_;
}

void Task::set_parent(Task* task) {
  parent_ = task;
}

void Task::set_brs_policy(int brs_policy) {
  brs_policy_ = brs_policy == brisbane_default ? platform_->device_default() : brs_policy;
  if (!HasSubtasks()) return;
  for (std::vector<Task*>::iterator I = subtasks_.begin(), E = subtasks_.end(); I != E; ++I)
    (*I)->set_brs_policy(brs_policy);
}

void Task::set_opt(const char* opt) {
  if (!opt) return;
  memset(opt_, 0, sizeof(opt_));
  strncpy(opt_, opt, strlen(opt));
}

void Task::AddCommand(Command* cmd) {
  if (ncmds_ == 63) _error("ncmds[%d]", ncmds_);
  cmds_[ncmds_++] = cmd;
  if (cmd->type() == BRISBANE_CMD_KERNEL) {
    if (cmd_kernel_) _error("kernel[%s] is already set", cmd->kernel()->name());
    if (!given_name_) strcpy(name_, cmd->kernel()->name());
    cmd_kernel_ = cmd;
  }
  if (!system_ &&
     (cmd->type() == BRISBANE_CMD_KERNEL || cmd->type() == BRISBANE_CMD_H2D ||
      cmd->type() == BRISBANE_CMD_H2DNP  || cmd->type() == BRISBANE_CMD_D2H))
    cmd_last_ = cmd;
}

void Task::ClearCommands() {
  for (int i = 0; i < ncmds_; i++) delete cmds_[i];
  ncmds_ = 0;
}

bool Task::Dispatchable() {
  for (int i = 0; i < ndepends_; i++) {
    if (depends_[i]->status() != BRISBANE_COMPLETE) return false;
  }
  return true;
}

bool Task::Executable() {
  pthread_mutex_lock(&mutex_executable_);
  if (status_ == BRISBANE_NONE) {
    status_ = BRISBANE_RUNNING;
    pthread_mutex_unlock(&mutex_executable_);
    return true;
  }
  pthread_mutex_unlock(&mutex_executable_);
  return false;
}

void Task::Complete() {
  pthread_mutex_lock(&mutex_complete_);
  status_ = BRISBANE_COMPLETE;
  pthread_cond_broadcast(&complete_cond_);
  pthread_mutex_unlock(&mutex_complete_);
  if (parent_) parent_->CompleteSub();
  else scheduler_->Invoke();
}

void Task::CompleteSub() {
  pthread_mutex_lock(&mutex_subtasks_);
  if (++subtasks_complete_ == subtasks_.size()) Complete();
  pthread_mutex_unlock(&mutex_subtasks_);
}

void Task::Wait() {
  pthread_mutex_lock(&mutex_complete_);
  if (status_ != BRISBANE_COMPLETE)
    pthread_cond_wait(&complete_cond_, &mutex_complete_);
  pthread_mutex_unlock(&mutex_complete_);
}

void Task::AddSubtask(Task* subtask) {
  subtask->set_parent(this);
  subtask->set_brs_policy(brs_policy_);
  subtasks_.push_back(subtask);
}

bool Task::HasSubtasks() {
  return !subtasks_.empty();
}

void Task::AddDepend(Task* task) {
  for (int i = 0; i < ndepends_; i++) if (task == depends_[i]) return;
  if (ndepends_ == 63) _error("ndepends[%d]", ndepends_);
  depends_[ndepends_++] = task;
}

void Task::Submit(int brs_policy, const char* opt, int sync) {
  set_brs_policy(brs_policy);
  set_opt(opt);
  sync_ = sync;
  cmd_last_->set_last();
}

} /* namespace rt */
} /* namespace brisbane */
