#include "WorkloadManager.h"
#include "Debug.h"
#include "Dependency.h"
#include "Device.h"
#include "Task.h"

namespace brisbane {
namespace rt {

WorkloadManager::WorkloadManager(Device* device) {
    device_ = device;
    device_->set_manager(this);
    dependency_ = new Dependency();
    queue_ = new LockFreeQueue<Task*>(1024);
}

WorkloadManager::~WorkloadManager() {
    delete dependency_;
    delete queue_;
}

void WorkloadManager::Enqueue(Task* task) {
    task->set_dev(device_);
    while (!queue_->Enqueue(task)) {}
    Invoke();
}

void WorkloadManager::Execute(Task* task) {
    if (!task->Executable()) return;
    dependency_->Resolve(task);
    device_->Execute(task);
    task->Complete();
}

void WorkloadManager::Run() {
    while (true) {
        Sleep();
        if (!running_) break;
        Task* task = NULL;
        while (queue_->Dequeue(&task)) Execute(task);
    }
}

} /* namespace rt */
} /* namespace brisbane */
