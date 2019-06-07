#ifndef BRISBANE_RT_SRC_POLICY_SPECIFIC_H
#define BRISBANE_RT_SRC_POLICY_SPECIFIC_H

#include "Policy.h"

namespace brisbane {
namespace rt {

class PolicySpecific : public Policy {
public:
    PolicySpecific(Scheduler* scheduler);
    virtual ~PolicySpecific();

    virtual void GetDevices(Task* task, Device** devs, int* ndevs);
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_POLICY_SPECIFIC_H */
