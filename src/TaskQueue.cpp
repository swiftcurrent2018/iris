#include "TaskQueue.h"

namespace brisbane {
namespace rt {

TaskQueue::TaskQueue() {
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
    for (std::list<Task*>::iterator it = tasks_.begin(); it != tasks_.end(); ++it) {
        Task* t = *it;
        if (!t->Submittable()) continue;
        if (t->marker() && it != tasks_.begin()) continue;
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
