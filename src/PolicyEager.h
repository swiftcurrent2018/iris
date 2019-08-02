#ifndef BRISBANE_RT_SRC_POLICY_EAGER_H
#define BRISBANE_RT_SRC_POLICY_EAGER_H

#include "Policy.h"

namespace brisbane {
namespace rt {

class PolicyEager : public Policy {
public:
    PolicyEager(Scheduler* scheduler);
    virtual ~PolicyEager();

    virtual void GetDevices(Task* task, Device** devs, int* ndevs);
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_POLICY_EAGER_H */
