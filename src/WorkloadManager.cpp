#include "WorkloadManager.h"
#include "Debug.h"
#include "Device.h"
#include "Task.h"

namespace brisbane {
namespace rt {

WorkloadManager::WorkloadManager(Device* device) {
    device_ = device;
    device_->set_manager(this);
    queue_ = new LockFreeQueue<Task*>(1024);
}

WorkloadManager::~WorkloadManager() {
    delete queue_;
}

void WorkloadManager::Enqueue(Task* task) {
    while (!queue_->Enqueue(task)) {}
    Invoke();
}

void WorkloadManager::Execute(Task* task) {
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
