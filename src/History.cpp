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
        times_avg_[i] = 0.0;
        cnts_[i] = 0;
    }
}

History::~History() {
}

void History::Add(Command* cmd, Device* dev, double time) {
    Kernel* kernel = cmd->kernel();
    int dev_no = dev->dev_no();
    int c = cnts_[dev_no];
    times_[dev_no] += time;
    cnts_[dev_no]++;
    times_avg_[dev_no] = (times_avg_[dev_no] * c + time) / (c + 1);
}

Device* History::OptimalDevice(Task* task) {
    for (int i = 0; i < ndevs_; i++) {
        if (cnts_[i] == 0) return platform_->device(i);
    }

    double min_time = times_avg_[0];
    int min_dev = 0;
    for (int i = 0; i < ndevs_; i++) {
        if (times_avg_[i] < min_time) {
            min_time = times_avg_[i];
            min_dev = i;
        }
    }
    return platform_->device(min_dev);
}

double History::time() {
    double time = 0.0;
    for (int i = 0; i < ndevs_; i++) {
        time += times_[i];
    }
    return time;
}

int History::cnt() {
    double cnt = 0.0;
    for (int i = 0; i < ndevs_; i++) {
        cnt += cnts_[i];
    }
    return cnt;
}

} /* namespace rt */
} /* namespace brisbane */
