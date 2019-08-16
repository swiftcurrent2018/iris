#include "Command.h"
#include "Task.h"

namespace brisbane {
namespace rt {

Command::Command(Task* task, int type) {
  task_ = task;
  type_ = type;
  time_ = 0.0;
  kernel_args_ = NULL;
}

Command::~Command() {
  if (kernel_args_) {
    for (std::map<int, KernelArg*>::iterator I = kernel_args_->begin(), E = kernel_args_->end(); I != E; ++I)
      delete I->second;
    delete kernel_args_;
  }
}

double Command::SetTime(double t) {
  if (time_ != 0.0) _error("double set time[%lf]", t);
  time_ = t;
  task_->TimeInc(t);
  return time_;
}

Command* Command::Create(Task* task, int type) {
  return new Command(task, type);
}

Command* Command::CreateBuild(Task* task) {
  return Create(task, BRISBANE_CMD_BUILD);
}

Command* Command::CreateKernel(Task* task, Kernel* kernel, int dim, size_t* off, size_t* ndr) {
  Command* cmd = Create(task, BRISBANE_CMD_KERNEL);
  cmd->kernel_ = kernel;
  cmd->kernel_args_ = kernel->ExportArgs();
  cmd->dim_ = dim;
  for (int i = 0; i < dim; i++) {
    cmd->off_[i] = off[i];
    cmd->ndr_[i] = ndr[i];
  }
  for (int i = dim; i < 3; i++) {
    cmd->off_[i] = 0;
    cmd->ndr_[i] = 1;
  }
  return cmd;
}

Command* Command::CreateH2D(Task* task, Mem* mem, size_t off, size_t size, void* host) {
  Command* cmd = Create(task, BRISBANE_CMD_H2D);
  cmd->mem_ = mem;
  cmd->off_[0] = off;
  cmd->size_ = size;
  cmd->host_ = host;
  return cmd;
}

Command* Command::CreateH2DNP(Task* task, Mem* mem, size_t off, size_t size, void* host) {
  Command* cmd = Create(task, BRISBANE_CMD_H2DNP);
  cmd->mem_ = mem;
  cmd->off_[0] = off;
  cmd->size_ = size;
  cmd->host_ = host;
  return cmd;
}

Command* Command::CreateD2H(Task* task, Mem* mem, size_t off, size_t size, void* host) {
  Command* cmd = Create(task, BRISBANE_CMD_D2H);
  cmd->mem_ = mem;
  cmd->off_[0] = off;
  cmd->size_ = size;
  cmd->host_ = host;
  return cmd;
}

Command* Command::CreatePresent(Task* task, Mem* mem, size_t off, size_t size, void* host) {
  Command* cmd = Create(task, BRISBANE_CMD_PRESENT);
  cmd->mem_ = mem;
  cmd->off_[0] = off;
  cmd->size_ = size;
  cmd->host_ = host;
  return cmd;
}

Command* Command::CreateReleaseMem(Task* task, Mem* mem) {
  Command* cmd = Create(task, BRISBANE_CMD_RELEASE_MEM);
  cmd->mem_ = mem;
  return cmd;
}

Command* Command::Duplicate(Command* cmd) {
  switch (cmd->type()) {
    case BRISBANE_CMD_KERNEL: return CreateKernel(cmd->task(), cmd->kernel(), cmd->dim(), cmd->off(), cmd->ndr());
    case BRISBANE_CMD_H2D:    return CreateH2D(cmd->task(), cmd->mem(), cmd->off(0), cmd->size(), cmd->host());
    case BRISBANE_CMD_D2H:    return CreateD2H(cmd->task(), cmd->mem(), cmd->off(0), cmd->size(), cmd->host());
  }
  return NULL;
}

void Command::Release(Command* cmd) {
  delete cmd;
}

} /* namespace rt */
} /* namespace brisbane */

