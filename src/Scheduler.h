#ifndef BRISBANE_RT_SRC_SCHEDULER_H
#define BRISBANE_RT_SRC_SCHEDULER_H

#include "Config.h"
#include "Thread.h"
//#include "Queue.h"

namespace brisbane {
namespace rt {

class Device;
class HubClient;
class Task;
class TaskQueue;
class Platform;
class Policies;
class Worker;

class Scheduler : public Thread {
public:
    Scheduler(Platform* platform);
    virtual ~Scheduler();

    void Enqueue(Task* task);

    Platform* platform() { return platform_; }
    Device** devices() { return devices_; }
    Worker** workers() { return workers_; }
    int ndevs() { return ndevs_; }
    int nworkers() { return ndevs_; }
    void CompleteTask(Task* task, Worker* worker);
    int RefreshNTasksOnDevs();
    size_t NTasksOnDev(int i);

private:
    void Submit(Task* task);
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
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_SCHEDULER_H */
