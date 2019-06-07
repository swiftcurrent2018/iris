#ifndef BRISBANE_RT_SRC_POLICY_RANDOM_H
#define BRISBANE_RT_SRC_POLICY_RANDOM_H

#include "Policy.h"

namespace brisbane {
namespace rt {

class PolicyRandom : public Policy {
public:
    PolicyRandom(Scheduler* scheduler);
    virtual ~PolicyRandom();

    virtual void GetDevices(Task* task, Device** devs, int* ndevs);

};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_POLICY_RANDOM_H */
