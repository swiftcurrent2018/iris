#include "PolicySpecific.h"
#include "Debug.h"
#include "Device.h"
#include "Task.h"

namespace brisbane {
namespace rt {

PolicySpecific::PolicySpecific(Scheduler* scheduler) {
    SetScheduler(scheduler);
    last_dev_no_ = 0;
}

PolicySpecific::~PolicySpecific() {
}

Device* PolicySpecific::GetDevice(Task* task) {
    int brs_device = task->brs_device();
    for (int i = last_dev_no_ + 1; i < last_dev_no_ + 1 + ndevs_; i++) {
        Device* dev = devices_[i % ndevs_];
        if ((dev->type() & brs_device) == dev->type() && dev->idle()) {
            last_dev_no_ = i % ndevs_;
            _debug("last_dev_no[%d]", last_dev_no_);
            return dev;
        }
    }
    for (int i = last_dev_no_ + 1; i < last_dev_no_ + 1 + ndevs_; i++) {
        Device* dev = devices_[i % ndevs_];
        if ((dev->type() & brs_device) == dev->type()) {
            last_dev_no_ = i % ndevs_;
            _debug("last_dev_no[%d]", last_dev_no_);
            return dev;
        }
    }
    return NULL;
}

} /* namespace rt */
} /* namespace brisbane */
