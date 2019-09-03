#include <Policy.h>
#include <Debug.h>
#include <Device.h>
#include <Scheduler.h>
#include <Task.h>

namespace brisbane {
namespace rt {

class PolicyFirst: public Policy {
public:
  PolicyFirst() {}
  virtual ~PolicyFirst() {}
  virtual void GetDevices(Task* task, Device** devs, int* ndevs) {
    _info("ndevs[%d]", ndevs_);
    devs[0] = devs_[0];
    *ndevs = 1;
  }
};

class PolicyLast : public Policy {
public:
  PolicyLast() {}
  virtual ~PolicyLast() {}
  virtual void GetDevices(Task* task, Device** devs, int* ndevs) {
    _info("ndevs[%d]", ndevs_);
    devs[0] = devs_[ndevs_ - 1];
    *ndevs = 1;
  }
};

} /* namespace rt */
} /* namespace brisbane */

REGISTER_CUSTOM_POLICY(PolicyLast, policy_last)
REGISTER_CUSTOM_POLICY(PolicyLast, policy_first)

