#include "PolicySpecific.h"
#include "Debug.h"
#include "Device.h"
#include "Task.h"

namespace brisbane {
namespace rt {

PolicySpecific::PolicySpecific(Scheduler* scheduler) {
    SetScheduler(scheduler);
}

PolicySpecific::~PolicySpecific() {
}

Device* PolicySpecific::GetDevice(Task* task) {
    int brs_device = task->brs_device();
    for (int i = 0; i < ndevs_; i++) {
        Device* dev = devices_[i];
        if (dev->type() == brs_device) return dev;
    }
    return NULL;
}

} /* namespace rt */
} /* namespace brisbane */
