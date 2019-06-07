#ifndef BRISBANE_RT_SRC_WORKLOAD_MANAGER_H
#define BRISBANE_RT_SRC_WORKLOAD_MANAGER_H

#include "Thread.h"
#include "Queue.h"

namespace brisbane {
namespace rt {

class Device;
class Task;

class WorkloadManager : public Thread {
public:
    WorkloadManager(Device* device);
    virtual ~WorkloadManager();

    void Enqueue(Task* task);

private:
    void Execute(Task* task);
    virtual void Run();

private:
    LockFreeQueue<Task*>* queue_;
    Device* device_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_WORKLOAD_MANAGER_H */

