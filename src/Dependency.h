#ifndef BRISBANE_RT_SRC_DEPENDENCY_H
#define BRISBANE_RT_SRC_DEPENDENCY_H

namespace brisbane {
namespace rt {

class Command;
class Task;

class Dependency {
public:
    Dependency();
    ~Dependency();

    void Resolve(Task* task);

private:
    void Resolve(Task* task, Command* cmd);

};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_DEPENDENCY_H */
