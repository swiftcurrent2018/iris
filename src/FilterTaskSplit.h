#ifndef BRISBANE_RT_SRC_FILTER_TASK_SPLIT_H
#define BRISBANE_RT_SRC_FILTER_TASK_SPLIT_H

#include "Filter.h"

namespace brisbane {
namespace rt {

class Polyhedral;
class Platform;

class FilterTaskSplit : public Filter {
public:
  FilterTaskSplit(Polyhedral* polyhedral, Platform* platform);
  virtual ~FilterTaskSplit();

  virtual int Execute(Task* task);

private:
  Polyhedral* polyhedral_;
  Platform* platform_;
};

} /* namespace rt */
} /* namespace brisbane */


#endif /* BRISBANE_RT_SRC_FILTER_TASK_SPLIT_H */

