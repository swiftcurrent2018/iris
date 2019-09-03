#include "Policies.h"
#include "Debug.h"
#include "LoaderPolicy.h"
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
  for (std::map<std::string, LoaderPolicy*>::iterator I = policy_customs_.begin(), E = policy_customs_.end(); I != E; ++I)
    delete I->second;
}

Policy* Policies::GetPolicy(int brs_policy, char* opt) {
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
  if (brs_policy == brisbane_custom) {
    if (policy_customs_.find(std::string(opt)) != policy_customs_.end()) {
      Policy* policy = policy_customs_[opt]->policy();
      policy->SetScheduler(scheduler_);
      return policy;
    }
  }
  _error("unknown policy [%d] [0x%x] [%s]", brs_policy, brs_policy, opt);
  return policy_any_;
}

int Policies::Register(const char* lib, const char* name) {
  LoaderPolicy* loader = new LoaderPolicy(lib, name);
  std::string namestr = std::string(name);
  if (policy_customs_.find(namestr) != policy_customs_.end()) {
    _error("existing policy name[%s]", name);
    return BRISBANE_ERR;
  }
  if (loader->Load() != BRISBANE_OK) {
    _error("cannot load custom policy[%s]", name);
    return BRISBANE_ERR;
  }
  _debug("lib[%s] name[%s]", lib, name);
  policy_customs_.insert(std::pair<std::string, LoaderPolicy*>(namestr, loader));
  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */

