#ifndef BRISBANE_RT_SRC_SCHEDULER_H
#define BRISBANE_RT_SRC_SCHEDULER_H

#include "Thread.h"
#include "Queue.h"
#include "Define.h"

namespace brisbane {
namespace rt {

class Device;
class Task;
class Platform;
class Policies;
class WorkloadManager;

class Scheduler : public Thread {
public:
    Scheduler(Platform* platform);
    virtual ~Scheduler();

    void Enqueue(Task* task);

    Platform* platform() { return platform_; }
    Device** devices() { return devices_; }
    int ndevs() { return ndevs_; }

private:
    void Execute(Task* task);
    virtual void Run();

    void InitWorkloadManagers();
    void DestroyWorkloadManagers();

private:
    LockFreeQueue<Task*>* queue_;
    Platform* platform_;

    Policies* policies_;
    Device** devices_;
    WorkloadManager* managers_[BRISBANE_MAX_NDEVS];
    int ndevs_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_SCHEDULER_H */
