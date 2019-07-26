#ifndef BRISBANE_RT_SRC_TASK_H
#define BRISBANE_RT_SRC_TASK_H

#include "Object.h"
#include "Command.h"
#include "Platform.h"
#include <pthread.h>
#include <vector>

#define BRISBANE_COMPLETE   0x0
#define BRISBANE_RUNNING    0x1
#define BRISBANE_SUBMITTED  0x2
#define BRISBANE_QUEUED     0x3
#define BRISBANE_NONE       0x4

#define BRISBANE_TASK       0x0
#define BRISBANE_MARKER     0x1

namespace brisbane {
namespace rt {

class Scheduler;

class Task: public Object<struct _brisbane_task, Task> {
public:
    Task(Platform* platform, int type = BRISBANE_TASK);
    virtual ~Task();

    void AddCommand(Command* cmd);

    void AddSubtask(Task* subtask);
    bool HasSubtasks();

    void AddDepend(Task* task);

    bool Submittable();
    bool Executable();
    void Complete();
    void Wait();

    int type() { return type_; }
    bool marker() { return type_ == BRISBANE_MARKER; }
    int status() { return status_; }
    Task* parent() { return parent_; }
    Command* cmd(int i) { return cmds_[i]; }
    Command* cmd_kernel() { return cmd_kernel_; }
    void set_dev(Device* dev) { dev_ = dev; }
    Device* dev() { return dev_; }
    int ncmds() { return ncmds_; }
    void set_parent(Task* task) { parent_ = task; }
    void set_brs_device(int brs_device);
    int brs_device() { return brs_device_; }
    std::vector<Task*>* subtasks() { return &subtasks_; }

private:
    void CompleteSub();

private:
    Task* parent_;
    int ncmds_;
    Command* cmds_[64];
    Command* cmd_kernel_;
    Device* dev_;
    Platform* platform_;
    Scheduler* scheduler_;
    std::vector<Task*> subtasks_;
    size_t subtasks_complete_;

    Task* depends_[64];
    int ndepends_;

    int brs_device_;

    int type_;
    int status_;
    pthread_mutex_t executable_mutex_;
    pthread_mutex_t complete_mutex_;
    pthread_cond_t complete_cond_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_TASK_H */
