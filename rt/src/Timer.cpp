#include "Timer.h"
#include <sys/time.h>
#include <time.h>

namespace brisbane {
namespace rt {

Timer::Timer() {
    base_sec_ = 0.0;
}

Timer::~Timer() {

}

double Timer::Now() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + 1.e-6 * t.tv_usec - base_sec_;
}

void Timer::Start() {
    start_ = Now();
}

double Timer::Stop() {
    return Now() - start_;
}

} /* namespace rt */
} /* namespace brisbane */
