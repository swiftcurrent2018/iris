#ifndef BRISBANE_RT_SRC_DEVICE_H
#define BRISBANE_RT_SRC_DEVICE_H

#include "Platform.h"
#include "Task.h"

namespace brisbane {
namespace rt {

class Device {
public:
    Device(cl_device_id cldev, cl_context clctx, int dev_no, int platform_no);
    ~Device();

    void BuildProgram();

    void Execute(Task* task);
    void ExecuteKernel(Command* cmd);
    void ExecuteH2D(Command* cmd);
    void ExecuteD2H(Command* cmd);

    void Wait();

    int dev_no() { return dev_no_; }
    int type() { return type_; }
    char* name() { return name_; }

private:
    cl_device_id cldev_;
    cl_context clctx_;
    cl_command_queue clcmdq_;
    cl_program clprog_;
    cl_device_type cltype_;
    cl_int clerr_;

    int dev_no_;
    int platform_no_;
    int type_;
    char vendor_[64];
    char name_[64];
    char version_[64];
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_DEVICE_H */
