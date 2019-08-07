#include "Policies.h"
#include "Debug.h"
#include "PolicyData.h"
#include "PolicyDefault.h"
#include "PolicyEager.h"
#include "PolicyProfile.h"
#include "PolicyRandom.h"
#include "PolicySpecific.h"
#include "Platform.h"

namespace brisbane {
namespace rt {

Policies::Policies(Scheduler* scheduler) {
  scheduler_ = scheduler;
  policy_default_  = new PolicyDefault(scheduler_);
  policy_data_     = new PolicyData(scheduler_);
  policy_eager_    = new PolicyEager(scheduler_);
  policy_profile_  = new PolicyProfile(scheduler_, this);
  policy_random_   = new PolicyRandom(scheduler_);
  policy_specific_ = new PolicySpecific(scheduler_);
}

Policies::~Policies() {
  delete policy_data_;
  delete policy_random_;
  delete policy_specific_;
}

Policy* Policies::GetPolicy(int brs_device) {
  if (brs_device &  brisbane_cpu    ||
      brs_device &  brisbane_nvidia ||
      brs_device &  brisbane_amd    ||
      brs_device &  brisbane_gpu    ||
      brs_device &  brisbane_phi    ||
      brs_device &  brisbane_fpga) return policy_specific_;
  if (brs_device == brisbane_data) return policy_data_;
  if (brs_device == brisbane_default) return policy_default_;
  if (brs_device == brisbane_eager) return policy_eager_;
  if (brs_device == brisbane_profile) return policy_profile_;
  if (brs_device == brisbane_random) return policy_random_;
  return policy_random_;
}

} /* namespace rt */
} /* namespace brisbane */
