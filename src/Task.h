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

namespace brisbane {
namespace rt {

class Task: public Object<struct _brisbane_task, Task> {
public:
    Task(Platform* platform);
    virtual ~Task();

    void AddCommand(Command* cmd);
    void Submit(int brs_device);
    void Execute();

    void AddSubtask(Task* subtask);
    bool HasSubtasks();

    void Complete();
    void Wait();

    Task* parent() { return parent_; }
    Command* cmd(int i) { return cmds_[i]; }
    Command* cmd_kernel() { return cmd_kernel_; }
    void set_dev(Device* dev) { dev_ = dev; }
    Device* dev() { return dev_; }
    int num_cmds() { return num_cmds_; }
    void set_brs_device(int brs_device) { brs_device_ = brs_device; }
    int brs_device() { return brs_device_; }

private:
    Task* parent_;
    int num_cmds_;
    Command* cmds_[64];
    Command* cmd_kernel_;
    Device* dev_;
    Platform* platform_;
    std::vector<Task*> subtasks_;
    int brs_device_;

    int status_;
    pthread_mutex_t complete_mutex_;
    pthread_cond_t complete_cond_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_TASK_H */
