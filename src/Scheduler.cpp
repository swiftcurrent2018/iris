#include "Scheduler.h"
#include "Debug.h"
#include "Device.h"
#include "HubClient.h"
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
  //queue_ = new LockFreeQueue<Task*>(1024);
  queue_ = new TaskQueue();
  hub_client_ = new HubClient(this);
  InitWorkers();
  InitHubClient();
}

Scheduler::~Scheduler() {
  DestroyWorkers();
  delete queue_;
  delete policies_;
  delete hub_client_;
}

void Scheduler::InitWorkers() {
  for (int i = 0; i < ndevs_; i++) {
    workers_[i] = new Worker(devices_[i], this);
    workers_[i]->Start();
  }
}

void Scheduler::DestroyWorkers() {
  for (int i = 0; i < ndevs_; i++) delete workers_[i];
}

void Scheduler::InitHubClient() {
  hub_available_ = hub_client_->Init() == BRISBANE_OK;
  _info("hub_available[%d]", hub_available_);
}

void Scheduler::CompleteTask(Task* task, Worker* worker) {
  int dev_no = worker->device()->dev_no();
  if (hub_available_) hub_client_->TaskDec(dev_no, 1);
}

int Scheduler::RefreshNTasksOnDevs() {
  if (!hub_available_) {
    for (int i = 0; i < ndevs_; i++) ntasks_on_devs_[i] = workers_[i]->ntasks();
    return BRISBANE_OK;
  }
  hub_client_->TaskAll(ntasks_on_devs_, ndevs_);
  return BRISBANE_OK;
}

size_t Scheduler::NTasksOnDev(int i) {
  return ntasks_on_devs_[i];
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
  if (ndevs == 0) {
    _error("no device[0x%x]", brs_device);
    task->Complete();
  }
  for (int i = 0; i < ndevs; i++) {
    devs[i]->worker()->Enqueue(task);
    if (hub_available_) hub_client_->TaskInc(devs[i]->dev_no(), 1);
  }
}

} /* namespace rt */
} /* namespace brisbane */
