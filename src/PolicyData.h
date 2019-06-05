#ifndef BRISBANE_RT_SRC_POLICY_DATA_H
#define BRISBANE_RT_SRC_POLICY_DATA_H

#include "Policy.h"

namespace brisbane {
namespace rt {

class PolicyData : public Policy {
public:
    PolicyData(Scheduler* scheduler);
    virtual ~PolicyData();

    virtual Device* GetDevice(Task* task);

};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_POLICY_DATA_H */
