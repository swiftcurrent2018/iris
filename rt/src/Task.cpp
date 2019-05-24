#include "Task.h"
#include "Debug.h"
#include "Device.h"
#include "Kernel.h"
#include "Mem.h"

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
    int target_brs_device = brs_device;
    if (brs_device == brisbane_device_auto) {
        _todo("device[0x%x]", brs_device);
        target_brs_device = brisbane_device_cpu;
    } else if (brs_device == brisbane_device_data) {
        int target_dev_no = GetDeviceData();
        dev_ = platform_->device(target_dev_no);
    } else {
        dev_ = platform_->AvailableDevice(target_brs_device);
    }
    platform_->ExecuteTask(this);
}

void Task::Execute() {
    dev_->Execute(this);
}

void Task::Wait() {
    dev_->Wait();
}

int Task::GetDeviceData() {
    size_t total_size[16];
    for (int i = 0; i < 16; i++) total_size[i] = 0UL;
    for (int i = 0; i < num_cmds_; i++) {
        Command* cmd = cmds_[i];
        if (cmd->type() == BRISBANE_CMD_KERNEL) {
            Kernel* kernel = cmd->kernel();
            std::map<int, KernelArg*> args = kernel->args();
            for (std::map<int, KernelArg*>::iterator it = args.begin(); it != args.end(); ++it) {
                Mem* mem = it->second->mem;
                if (!mem || !mem->owner()) continue;
                total_size[mem->owner()->dev_no()] += mem->size();
            }
        }
    }
    int target_dev = 0;
    size_t max_size = 0;
    for (int i = 0; i < 16; i++) {
        if (total_size[i] > max_size) {
            max_size = total_size[i];
            target_dev = i;
        }
    }
    return target_dev;
}

} /* namespace rt */
} /* namespace brisbane */
