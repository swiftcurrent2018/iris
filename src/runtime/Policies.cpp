#include "Policies.h"
#include "Debug.h"
#include "PolicyAll.h"
#include "PolicyAny.h"
#include "PolicyData.h"
#include "PolicyDefault.h"
#include "PolicyDevice.h"
#include "PolicyProfile.h"
#include "PolicyRandom.h"
#include "Platform.h"

namespace brisbane {
namespace rt {

Policies::Policies(Scheduler* scheduler) {
  scheduler_ = scheduler;
  policy_all_      = new PolicyAll(scheduler_);
  policy_any_      = new PolicyAny(scheduler_);
  policy_data_     = new PolicyData(scheduler_);
  policy_default_  = new PolicyDefault(scheduler_);
  policy_device_   = new PolicyDevice(scheduler_);
  policy_profile_  = new PolicyProfile(scheduler_, this);
  policy_random_   = new PolicyRandom(scheduler_);
}

Policies::~Policies() {
  delete policy_all_;
  delete policy_any_;
  delete policy_data_;
  delete policy_default_;
  delete policy_device_;
  delete policy_profile_;
  delete policy_random_;
}

Policy* Policies::GetPolicy(int brs_policy) {
  if (brs_policy &  brisbane_cpu    ||
      brs_policy &  brisbane_nvidia ||
      brs_policy &  brisbane_amd    ||
      brs_policy &  brisbane_gpu    ||
      brs_policy &  brisbane_phi    ||
      brs_policy &  brisbane_fpga)    return policy_device_;
  if (brs_policy == brisbane_all)     return policy_all_;
  if (brs_policy == brisbane_any)     return policy_any_;
  if (brs_policy == brisbane_data)    return policy_data_;
  if (brs_policy == brisbane_default) return policy_default_;
  if (brs_policy == brisbane_profile) return policy_profile_;
  if (brs_policy == brisbane_random)  return policy_random_;
  _error("unknown policy [%d] [0x%x]", brs_policy, brs_policy);
  return policy_any_;
}

} /* namespace rt */
} /* namespace brisbane */
