#ifndef BRISBANE_RT_SRC_FILTER_H
#define BRISBANE_RT_SRC_FILTER_H

namespace brisbane {
namespace rt {

class Task;

class Filter {
public:
  Filter() {}
  virtual ~Filter() {}

  virtual int Execute(Task* task) = 0;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_FILTER_H */
