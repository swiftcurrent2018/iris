#ifndef BRISBANE_RT_SRC_POLICY_H
#define BRISBANE_RT_SRC_POLICY_H

namespace brisbane {
namespace rt {

class Device;
class Scheduler;
class Task;

class Policy {
public:
    Policy();
    virtual ~Policy();

    virtual void GetDevices(Task* task, Device** devs, int* ndevs) = 0;

protected:
    void SetScheduler(Scheduler* scheduler);

protected:
    Scheduler* scheduler_;
    Device** devices_;
    int ndevs_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_POLICY_H */
