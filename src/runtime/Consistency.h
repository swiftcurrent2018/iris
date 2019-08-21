#ifndef BRISBANE_RT_SRC_CONSISTENCY_H
#define BRISBANE_RT_SRC_CONSISTENCY_H

#include <brisbane/brisbane_poly_types.h>
#include "Kernel.h"

namespace brisbane {
namespace rt {

class Command;
class Mem;
class Task;

class Consistency {
public:
  Consistency();
  ~Consistency();

  void Resolve(Task* task);

private:
  void Resolve(Task* task, Command* cmd);
  void ResolveWithPolymem(Task* task, Command* cmd, Mem* mem, KernelArg* arg, brisbane_poly_mem* polymem);
  void ResolveWithoutPolymem(Task* task, Command* cmd, Mem* mem);
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_CONSISTENCY_H */
