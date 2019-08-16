#include "Scheduler.h"
#include "Debug.h"
#include "Device.h"
#include "HubClient.h"
#include "Platform.h"
#include "Policies.h"
#include "Policy.h"
#include "Profiler.h"
#include "Task.h"
#include "TaskQueue.h"
#include "Timer.h"
#include "Worker.h"

namespace brisbane {
namespace rt {

Scheduler::Scheduler(Platform* platform) {
  platform_ = platform;
  devices_ = platform_->devices();
  ndevs_ = platform_->ndevs();
  enable_profiler_ = platform->enable_profiler();
  nprofilers_ = platform->nprofilers();
  profilers_ = platform->profilers();
  pthread_mutex_init(&mutex_, NULL);
  timer_ = new Timer();
  policies_ = new Policies(this);
  //queue_ = new LockFreeQueue<Task*>(1024);
  queue_ = new TaskQueue(this);
  hub_client_ = new HubClient(this);
  InitWorkers();
  InitHubClient();
}

Scheduler::~Scheduler() {
  DestroyWorkers();
  delete queue_;
  delete policies_;
  delete hub_client_;
  pthread_mutex_destroy(&mutex_);
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
}

void Scheduler::StartTask(Task* task, Worker* worker) {
  task->set_time_start(timer_->Now());
}

void Scheduler::CompleteTask(Task* task, Worker* worker) {
  Device* dev = worker->device();
  int devno = dev->devno();
  if (hub_available_) hub_client_->TaskDec(devno, 1);
  if (enable_profiler_ & !task->system()) {
    task->set_time_end(timer_->Now());
    pthread_mutex_lock(&mutex_); //TODO: no lock
    for (int i = 0; i < nprofilers_; i++) profilers_[i]->CompleteTask(task); 
    pthread_mutex_unlock(&mutex_);
  }
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
  while (!queue_->Enqueue(task)) {}
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
    for (int i = 0; i < ndevs_; i++) workers_[i]->Enqueue(task->subtask(i));
    return;
  }
  if (!task->HasSubtasks()) {
    SubmitWorker(task);
    return;
  }
  std::vector<Task*>* subtasks = task->subtasks();
  for (std::vector<Task*>::iterator I = subtasks->begin(), E = subtasks->end(); I != E; ++I)
    SubmitWorker(*I);
}

void Scheduler::SubmitWorker(Task* task) {
  int brs_policy = task->brs_policy();
  int ndevs = 0;
  Device* devs[BRISBANE_MAX_NDEVS];
  policies_->GetPolicy(brs_policy)->GetDevices(task, devs, &ndevs);
  if (ndevs == 0) {
    _error("no device for policy[0x%x]", brs_policy);
    task->Complete();
  }
  for (int i = 0; i < ndevs; i++) {
    devs[i]->worker()->Enqueue(task);
    if (hub_available_) hub_client_->TaskInc(devs[i]->devno(), 1);
  }
}

} /* namespace rt */
} /* namespace brisbane */
