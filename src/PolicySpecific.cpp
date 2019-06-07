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

void PolicySpecific::GetDevices(Task* task, Device** devs, int* ndevs) {
    int brs_device = task->brs_device();
    int n = 0;
    for (int i = 0; i < ndevs_; i++) {
        Device* dev = devices_[i];
        if ((dev->type() & brs_device) == dev->type()) {
            devs[n++] = dev;
        }
    }
    *ndevs = n;
}

} /* namespace rt */
} /* namespace brisbane */
