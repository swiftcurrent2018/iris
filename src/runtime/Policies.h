#ifndef BRISBANE_SRC_RT_POLICIES_H
#define BRISBANE_SRC_RT_POLICIES_H

namespace brisbane {
namespace rt {

class Policy;
class Scheduler;

class Policies {
public:
  Policies(Scheduler* scheduler);
  ~Policies();

  Policy* GetPolicy(int brs_policy);

private:
  Scheduler* scheduler_;

  Policy* policy_all_;
  Policy* policy_any_;
  Policy* policy_data_;
  Policy* policy_default_;
  Policy* policy_device_;
  Policy* policy_profile_;
  Policy* policy_random_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_POLICIES_H */

