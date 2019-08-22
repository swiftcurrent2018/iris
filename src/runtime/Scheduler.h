#ifndef BRISBANE_SRC_RT_SCHEDULER_H
#define BRISBANE_SRC_RT_SCHEDULER_H

#include "Config.h"
#include "Thread.h"
//#include "Queue.h"
#include <pthread.h>

namespace brisbane {
namespace rt {

class Device;
class HubClient;
class Task;
class TaskQueue;
class Timer;
class Platform;
class Policies;
class Profiler;
class Worker;

class Scheduler : public Thread {
public:
  Scheduler(Platform* platform);
  virtual ~Scheduler();

  void Enqueue(Task* task);

  Platform* platform() { return platform_; }
  Device** devices() { return devices_; }
  Worker** workers() { return workers_; }
  Worker* worker(int i) { return workers_[i]; }
  int ndevs() { return ndevs_; }
  int nworkers() { return ndevs_; }
  void StartTask(Task* task, Worker* worker);
  void CompleteTask(Task* task, Worker* worker);
  bool hub_available() { return hub_available_; }
  bool enable_profiler() { return enable_profiler_; }
  int RefreshNTasksOnDevs();
  size_t NTasksOnDev(int i);

private:
  void Submit(Task* task);
  void SubmitWorker(Task* task);
  virtual void Run();

  void InitWorkers();
  void DestroyWorkers();

  void InitHubClient();

private:
  //    LockFreeQueue<Task*>* queue_;
  TaskQueue* queue_;
  Platform* platform_;

  Policies* policies_;
  Device** devices_;
  Worker* workers_[BRISBANE_MAX_NDEVS];
  size_t ntasks_on_devs_[BRISBANE_MAX_NDEVS];
  int ndevs_;
  HubClient* hub_client_;
  bool hub_available_;
  bool enable_profiler_;
  int nprofilers_;
  Profiler** profilers_;
  Timer* timer_;
  pthread_mutex_t mutex_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_SCHEDULER_H */
