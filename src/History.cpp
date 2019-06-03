#include "History.h"
#include "Debug.h"
#include "Command.h"
#include "Device.h"
#include "Platform.h"
#include "Kernel.h"

namespace brisbane {
namespace rt {

History::History(Kernel* kernel) {
    kernel_ = kernel;
    platform_ = kernel_->platform();
    ndevs_ = platform_->ndevs();
    for (int i = 0; i < ndevs_; i++) {
        times_[i] = 0.0;
        cnts_[i] = 0;
    }
}

History::~History() {
}

void History::Add(Command* cmd, Device* dev, double time) {
    Kernel* kernel = cmd->kernel();
    int dev_no = dev->dev_no();
    int c = cnts_[dev_no];
    double t = times_[dev_no];
    cnts_[dev_no]++;
    times_[dev_no] = (t * c + time) / (c + 1);
}

Device* History::OptimalDevice(Task* task) {
    for (int i = 0; i < ndevs_; i++) {
        if (cnts_[i] == 0) return platform_->device(i);
    }

    double min_time = times_[0];
    int min_dev = 0;
    for (int i = 0; i < ndevs_; i++) {
        if (times_[i] < min_time) {
            min_time = times_[i];
            min_dev = i;
        }
    }
    return platform_->device(min_dev);
}

} /* namespace rt */
} /* namespace brisbane */
