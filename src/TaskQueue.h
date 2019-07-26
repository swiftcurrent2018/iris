#ifndef BRISBANE_RT_SRC_TASK_QUEUE_H
#define BRISBANE_RT_SRC_TASK_QUEUE_H

#include "Task.h"
#include "Queue.h"
#include <pthread.h>
#include <list>

namespace brisbane {
namespace rt {

class TaskQueue {
public:
    TaskQueue();
    ~TaskQueue();

    bool Peek(Task** task);
    bool Enqueue(Task* task);
    bool Dequeue(Task** task);
    bool Empty();

private:
    std::list<Task*> tasks_;
    pthread_mutex_t mutex_tasks_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_TASK_QUEUE_H */
