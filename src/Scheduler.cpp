#include "Scheduler.h"
#include "Debug.h"
#include "Device.h"
#include "Platform.h"
#include "Policies.h"
#include "Policy.h"
#include "Task.h"

namespace brisbane {
namespace rt {

Scheduler::Scheduler(Platform* platform) {
    platform_ = platform;
    devices_ = platform_->devices();
    ndevs_ = platform_->ndevs();
    policies_ = new Policies(this);
    queue_ = new LockFreeQueue<Task*>(1024);
}

Scheduler::~Scheduler() {
    delete policies_;
}

void Scheduler::Enqueue(Task* task) {
    while (!queue_->Enqueue(task)) {}
    Invoke();
}

void Scheduler::Run() {
    while (true) {
        sem_wait(&sem_);
        if (!running_) break;
        Task* task = NULL;
        while (queue_->Dequeue(&task)) Execute(task);
    }
}

void Scheduler::Execute(Task* task) {
    Device* dev = AvailableDevice(task);
    task->set_dev(dev);
    platform_->ExecuteTask(task);
    task->Complete();
}

Device* Scheduler::AvailableDevice(Task* task) {
    int brs_device = task->brs_device();
    Policy* policy = policies_->GetPolicy(brs_device);
    Device* dev = policy->GetDevice(task);
}

} /* namespace rt */
} /* namespace brisbane */
