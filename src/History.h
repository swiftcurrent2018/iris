#ifndef BRISBANE_RT_SRC_HISTORY_H
#define BRISBANE_RT_SRC_HISTORY_H

#include <map>
#include <string>

namespace brisbane {
namespace rt {

class Command;
class Device;
class Kernel;
class Platform;
class Task;

class History {
public:
    History(Kernel* kernel);
    ~History();

    void Add(Command* cmd, Device* dev, double time);
    Device* OptimalDevice(Task* task);

    double time();
    int cnt();

private:
    Kernel* kernel_;
    Platform* platform_;
    int ndevs_;
    double times_[16];
    double times_avg_[16];
    int cnts_[16];
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_HISTORY_H */
