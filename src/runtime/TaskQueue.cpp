#include "TaskQueue.h"

namespace brisbane {
namespace rt {

TaskQueue::TaskQueue() {
  last_sync_task_ = NULL;
  pthread_mutex_init(&mutex_tasks_, NULL);
}

TaskQueue::~TaskQueue() {
  pthread_mutex_destroy(&mutex_tasks_);
}

bool TaskQueue::Peek(Task** task) {
  pthread_mutex_lock(&mutex_tasks_);
  if (tasks_.empty()) {
    pthread_mutex_unlock(&mutex_tasks_);
    return false;
  }
  for (std::list<Task*>::iterator I = tasks_.begin(), E = tasks_.end(); I != E; ++I) {
    Task* t = *I;
    if (!t->Submittable()) continue;
    if (t->marker() && I != tasks_.begin()) continue;
    *task = t;
    pthread_mutex_unlock(&mutex_tasks_);
    return true;
  }
  pthread_mutex_unlock(&mutex_tasks_);
  return false;
}

bool TaskQueue::Enqueue(Task* task) {
  pthread_mutex_lock(&mutex_tasks_);
  tasks_.push_back(task);
  if (last_sync_task_) task->AddDepend(last_sync_task_);
  if (last_sync_task_ && task->sync()) last_sync_task_->Release();
  if (task->sync()) {
    last_sync_task_ = task;
    last_sync_task_->Retain();
  }
  pthread_mutex_unlock(&mutex_tasks_);
  return true;
}

bool TaskQueue::Dequeue(Task** task) {
  pthread_mutex_lock(&mutex_tasks_);
  tasks_.remove(*task);
  pthread_mutex_unlock(&mutex_tasks_);
  return true;
}

bool TaskQueue::Empty() {
  bool empty = false;
  pthread_mutex_lock(&mutex_tasks_);
  empty = tasks_.empty();
  pthread_mutex_unlock(&mutex_tasks_);
  return empty;
}

} /* namespace rt */
} /* namespace brisbane */
