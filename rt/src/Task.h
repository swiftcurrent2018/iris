#ifndef BRISBANE_RT_SRC_TASK_H
#define BRISBANE_RT_SRC_TASK_H

#include "Object.h"
#include "Command.h"
#include "Platform.h"

namespace brisbane {
namespace rt {

class Task: public Object<struct _brisbane_task, Task> {
public:
    Task();
    virtual ~Task();

    void Add(Command* cmd);
    void Submit(int brs_device);
    void Execute();
    void Wait();

    Command* cmd(int i) { return cmds_[i]; }
    Command* cmd_kernel() { return cmd_kernel_; }
    Device* dev() { return dev_; }
    int num_cmds() { return num_cmds_; }

private:
    int num_cmds_;
    Command* cmds_[64];
    Command* cmd_kernel_;
    Device* dev_;
    Platform* platform_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_TASK_H */
