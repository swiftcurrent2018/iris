#include "Task.h"
#include "Debug.h"
#include "Device.h"

namespace brisbane {
namespace rt {

Task::Task() {
    num_cmds_ = 0;
    platform_ = Platform::GetPlatform();
}

Task::~Task() {
}

void Task::Add(Command* cmd) {
    cmds_[num_cmds_++] = cmd;
}

void Task::Submit(int brs_device) {
    Device* dev = platform_->AvailableDevice(brs_device);
    dev_ = dev;
    dev->Execute(this);
}

void Task::Wait() {
    dev_->Wait();
}

} /* namespace rt */
} /* namespace brisbane */
