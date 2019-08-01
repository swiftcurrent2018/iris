#ifndef BRISBANE_RT_SRC_WORKER_H
#define BRISBANE_RT_SRC_WORKER_H

#include "Thread.h"
#include "Queue.h"

namespace brisbane {
namespace rt {

class Consistency;
class Device;
class Task;

class Worker : public Thread {
public:
    Worker(Device* device);
    virtual ~Worker();

    void Enqueue(Task* task);

private:
    void Execute(Task* task);
    virtual void Run();

private:
    LockFreeQueue<Task*>* queue_;
    Consistency* consistency_;
    Device* device_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_WORKER_H */

