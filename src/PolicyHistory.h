#ifndef BRISBANE_RT_SRC_POLICY_HISTORY_H
#define BRISBANE_RT_SRC_POLICY_HISTORY_H

#include "Policy.h"

namespace brisbane {
namespace rt {

class Policies;

class PolicyHistory : public Policy {
public:
    PolicyHistory(Scheduler* scheduler, Policies* policies);
    virtual ~PolicyHistory();

    virtual void GetDevices(Task* task, Device** devs, int* ndevs);

private:
    Policies* policies_;

};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_POLICY_HISTORY_H */
