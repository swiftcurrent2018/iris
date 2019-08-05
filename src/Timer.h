#ifndef BRISBANE_RT_SRC_TIMER_H
#define BRISBANE_RT_SRC_TIMER_H

#define BRISBANE_TIMER_MAX      128
#define BRISBANE_TIMER_APP      1
#define BRISBANE_TIMER_INIT     2

#include <stddef.h>

namespace brisbane {
namespace rt {

class Timer {
public:
  Timer();
  ~Timer();

  double Now();
  double Start(int i);
  double Stop(int i);
  double Total(int i);

  size_t Inc(int i);
  size_t Inc(int i, size_t s);

private:
  double start_[BRISBANE_TIMER_MAX];
  double total_[BRISBANE_TIMER_MAX];
  size_t total_ul_[BRISBANE_TIMER_MAX];

  double base_sec_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_TIMER_H */
