#include "Worker.h"
#include "Debug.h"
#include "Consistency.h"
#include "Device.h"
#include "Scheduler.h"
#include "Task.h"

namespace brisbane {
namespace rt {

Worker::Worker(Device* device, Scheduler* scheduler) {
  device_ = device;
  scheduler_ = scheduler;
  device_->set_worker(this);
  consistency_ = new Consistency();
  queue_ = new LockFreeQueue<Task*>(1024);
  busy_ = false;
}

Worker::~Worker() {
  delete consistency_;
  delete queue_;
}

void Worker::Enqueue(Task* task) {
  while (!queue_->Enqueue(task)) {}
  Invoke();
}

void Worker::Execute(Task* task) {
  if (!task->Executable()) return;
  if (task->marker()) {
    task->Complete();
    return;
  }
  busy_ = true;
  scheduler_->StartTask(task, this);
  consistency_->Resolve(task);
  device_->Execute(task);
  task->Complete();
  scheduler_->CompleteTask(task, this);
  busy_ = false;
}

void Worker::Run() {
  while (true) {
    Sleep();
    if (!running_) break;
    Task* task = NULL;
    while (queue_->Dequeue(&task)) Execute(task);
  }
}

unsigned long Worker::ntasks() {
  return queue_->Size() + busy_ ? 1 : 0;
}

} /* namespace rt */
} /* namespace brisbane */
