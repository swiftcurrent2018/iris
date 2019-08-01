#include "Scheduler.h"
#include "Debug.h"
#include "Device.h"
#include "Platform.h"
#include "Policies.h"
#include "Policy.h"
#include "Task.h"
#include "TaskQueue.h"
#include "Worker.h"

namespace brisbane {
namespace rt {

Scheduler::Scheduler(Platform* platform) {
    platform_ = platform;
    devices_ = platform_->devices();
    ndevs_ = platform_->ndevs();
    policies_ = new Policies(this);
//    queue_ = new LockFreeQueue<Task*>(1024);
    queue_ = new TaskQueue();
    InitWorkers();
}

Scheduler::~Scheduler() {
    DestroyWorkers();
    delete queue_;
    delete policies_;
}

void Scheduler::InitWorkers() {
    for (int i = 0; i < ndevs_; i++) {
        workers_[i] = new Worker(devices_[i]);
        workers_[i]->Start();
    }
}

void Scheduler::DestroyWorkers() {
    for (int i = 0; i < ndevs_; i++) delete workers_[i];
}

void Scheduler::Enqueue(Task* task) {
    if (task->HasSubtasks()) {
        std::vector<Task*>* subtasks = task->subtasks();
        for (std::vector<Task*>::iterator it = subtasks->begin(); it != subtasks->end(); ++it) {
            while (!queue_->Enqueue(*it)) {}
        }
    } else while (!queue_->Enqueue(task)) {}
    Invoke();
}

void Scheduler::Run() {
    while (true) {
        Sleep();
        if (!running_) break;
        Task* task = NULL;
        while (queue_->Peek(&task)) {
            Submit(task);
            queue_->Dequeue(&task);
        }
    }
}

void Scheduler::Submit(Task* task) {
    if (task->marker()) {
        task->Complete();
        return;
    }
    int brs_device = task->brs_device();
    int ndevs = 0;
    Device* devs[BRISBANE_MAX_NDEVS];
    policies_->GetPolicy(brs_device)->GetDevices(task, devs, &ndevs);
    for (int i = 0; i < ndevs; i++) {
        devs[i]->worker()->Enqueue(task);
    }
}

} /* namespace rt */
} /* namespace brisbane */
