#include "Timer.h"
#include <sys/time.h>
#include <time.h>

namespace brisbane {
namespace rt {

Timer::Timer() {
  for (int i = 0; i < BRISBANE_TIMER_MAX; i++) {
    total_[i] = 0.0;
    total_ul_[i] = 0UL;
  }
  struct timeval t;
  gettimeofday(&t, NULL);
  base_sec_ = t.tv_sec;
  base_ms_ = t.tv_usec;
}

Timer::~Timer() {

}

double Timer::Now() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + 1.e-6 * t.tv_usec - base_sec_;
}

unsigned long Timer::NowUS() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_usec - base_ms_;
}

double Timer::Start(int i) {
  double t = Now();
  start_[i] = t;
  return t;
}

double Timer::Stop(int i) {
  double t = Now() - start_[i];
  total_[i] += t;
  return t;
}

double Timer::Total(int i) {
  return total_[i];
}

size_t Timer::Inc(int i) {
  return Inc(i, 1UL);
}

size_t Timer::Inc(int i, size_t s) {
  total_ul_[i] += s;
  return total_ul_[i];
}

} /* namespace rt */
} /* namespace brisbane */
