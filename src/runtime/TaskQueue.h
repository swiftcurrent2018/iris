#ifndef BRISBANE_SRC_RT_TASK_QUEUE_H
#define BRISBANE_SRC_RT_TASK_QUEUE_H

#include "Task.h"
#include "Queue.h"
#include <pthread.h>
#include <list>

namespace brisbane {
namespace rt {

class Scheduler;

class TaskQueue {
public:
  TaskQueue(Scheduler* scheduler);
  ~TaskQueue();

  bool Peek(Task** task);
  bool Enqueue(Task* task);
  bool Dequeue(Task** task);
  bool Empty();

private:
  Scheduler* scheduler_;
  std::list<Task*> tasks_;
  pthread_mutex_t mutex_tasks_;
  Task* last_sync_task_;
  bool enable_profiler_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_TASK_QUEUE_H */
