#include "PolicyRandom.h"
#include "Debug.h"
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

namespace brisbane {
namespace rt {

PolicyRandom::PolicyRandom(Scheduler* scheduler) {
    SetScheduler(scheduler);
    srand(time(NULL));
}

PolicyRandom::~PolicyRandom() {
}

Device* PolicyRandom::GetDevice(Task* task) {
    return devices_[rand() % ndevs_];
}

} /* namespace rt */
} /* namespace brisbane */
