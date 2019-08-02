#ifndef BRISBANE_RT_SRC_POLICIES_H
#define BRISBANE_RT_SRC_POLICIES_H

namespace brisbane {
namespace rt {

class Policy;
class Scheduler;

class Policies {
public:
    Policies(Scheduler* scheduler);
    ~Policies();

    Policy* GetPolicy(int brs_device);

private:
    Scheduler* scheduler_;

    Policy* policy_data_;
    Policy* policy_default_;
    Policy* policy_eager_;
    Policy* policy_profile_;
    Policy* policy_random_;
    Policy* policy_specific_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_POLICIES_H */

