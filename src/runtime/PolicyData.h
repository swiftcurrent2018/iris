#ifndef BRISBANE_RT_SRC_POLICY_DATA_H
#define BRISBANE_RT_SRC_POLICY_DATA_H

#include "Policy.h"

namespace brisbane {
namespace rt {

class PolicyData : public Policy {
public:
  PolicyData(Scheduler* scheduler);
  virtual ~PolicyData();

  virtual void GetDevices(Task* task, Device** devs, int* ndevs);

};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_POLICY_DATA_H */
