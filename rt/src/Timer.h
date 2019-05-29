#ifndef BRISBANE_RT_SRC_TIMER_H
#define BRISBANE_RT_SRC_TIMER_H

namespace brisbane {
namespace rt {

class Timer {
public:
    Timer();
    ~Timer();

    double Now();
    void Start();
    double Stop();

private:
    double base_sec_;
    double start_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_TIMER_H */
