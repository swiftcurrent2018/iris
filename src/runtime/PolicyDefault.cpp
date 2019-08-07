#include "PolicyDefault.h"
#include "Debug.h"
#include "Platform.h"

namespace brisbane {
namespace rt {

PolicyDefault::PolicyDefault(Scheduler* scheduler) {
  SetScheduler(scheduler);
}

PolicyDefault::~PolicyDefault() {
}

void PolicyDefault::GetDevices(Task* task, Device** devs, int* ndevs) {
  _error("cannot be here [%p]", ndevs);
  devs[0] = devices_[0];
  *ndevs = 1;
}

} /* namespace rt */
} /* namespace brisbane */
