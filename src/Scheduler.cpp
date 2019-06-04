#include "Scheduler.h"
#include "Debug.h"
#include "Platform.h"
#include "Task.h"

namespace brisbane {
namespace rt {

Scheduler::Scheduler(Platform* platform) {
    _check();
    platform_ = platform;
    queue_ = new LockFreeQueue<Task*>(1024);
}

Scheduler::~Scheduler() {
    _check();
}

void Scheduler::Enqueue(Task* task) {
    while (!queue_->Enqueue(task)) {}
    Invoke();
}

void Scheduler::Run() {
    _check();
    while (true) {
        sem_wait(&sem_);
        if (!running_) break;
        Task* task = NULL;
        while (queue_->Dequeue(&task)) Execute(task);
    }
}

void Scheduler::Execute(Task* task) {
    _check();
    task->Complete();
}

} /* namespace rt */
} /* namespace brisbane */
