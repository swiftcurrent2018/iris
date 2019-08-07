#ifndef BRISBANE_RT_SRC_CONSISTENCY_H
#define BRISBANE_RT_SRC_CONSISTENCY_H

namespace brisbane {
namespace rt {

class Command;
class Task;

class Consistency {
public:
  Consistency();
  ~Consistency();

  void Resolve(Task* task);

private:
  void Resolve(Task* task, Command* cmd);

};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_CONSISTENCY_H */
