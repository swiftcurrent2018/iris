#ifndef BRISBANE_RT_SRC_SCHEDULER_H
#define BRISBANE_RT_SRC_SCHEDULER_H

#include "Thread.h"
#include "Queue.h"

namespace brisbane {
namespace rt {

class Task;
class Platform;

class Scheduler : public Thread {
public:
    Scheduler(Platform* platform);
    virtual ~Scheduler();

    void Enqueue(Task* task);

private:
    void Execute(Task* task);
    virtual void Run();

private:
    LockFreeQueue<Task*>* queue_;
    Platform* platform_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_SCHEDULER_H */
