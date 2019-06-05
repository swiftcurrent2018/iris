#include "PolicyDefault.h"
#include "Debug.h"

namespace brisbane {
namespace rt {

PolicyDefault::PolicyDefault(Scheduler* scheduler) {
    SetScheduler(scheduler);
}

PolicyDefault::~PolicyDefault() {
}

Device* PolicyDefault::GetDevice(Task* task) {
    return devices_[0];
}

} /* namespace rt */
} /* namespace brisbane */
