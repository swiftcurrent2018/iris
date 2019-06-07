#include "PolicyDefault.h"
#include "Debug.h"

namespace brisbane {
namespace rt {

PolicyDefault::PolicyDefault(Scheduler* scheduler) {
    SetScheduler(scheduler);
}

PolicyDefault::~PolicyDefault() {
}

void PolicyDefault::GetDevices(Task* task, Device** devs, int* ndevs) {
    devs[0] = devices_[0];
    *ndevs = 1;
}

} /* namespace rt */
} /* namespace brisbane */
