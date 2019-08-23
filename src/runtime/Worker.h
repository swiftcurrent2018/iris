#ifndef BRISBANE_SRC_RT_WORKER_H
#define BRISBANE_SRC_RT_WORKER_H

#include "Thread.h"
#include "Queue.h"

namespace brisbane {
namespace rt {

class Consistency;
class Device;
class Scheduler;
class Task;

class Worker : public Thread {
public:
  Worker(Device* dev, Scheduler* scheduler);
  virtual ~Worker();

  void Enqueue(Task* task);
  bool busy() { return busy_; }
  unsigned long ntasks();
  Device* device() { return dev_; }

private:
  void Execute(Task* task);
  virtual void Run();

private:
  LockFreeQueue<Task*>* queue_;
  Consistency* consistency_;
  Device* dev_;
  Scheduler* scheduler_;
  bool busy_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_WORKER_H */
