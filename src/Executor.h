#ifndef BRISBANE_RT_SRC_EXECUTOR_H
#define BRISBANE_RT_SRC_EXECUTOR_H

namespace brisbane {
namespace rt {

class Command;
class Task;

class Executor {
public:
    Executor();
    ~Executor();

    void Execute(Task* task);

private:
    void ResolveDependency(Command* cmd, Task* task);
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_EXECUTOR_H */
