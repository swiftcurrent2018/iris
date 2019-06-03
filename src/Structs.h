#ifndef BRISBANE_RT_SRC_STRUCTS_H
#define BRISBANE_RT_SRC_STRUCTS_H

namespace brisbane {
namespace rt {
class Kernel;
class Mem;
class Task;
} /* namespace rt */
} /* namespace brisbane */

struct _brisbane_task {
    brisbane::rt::Task* class_obj;
};

struct _brisbane_kernel {
    brisbane::rt::Kernel* class_obj;
};

struct _brisbane_mem {
    brisbane::rt::Mem* class_obj;
};

#endif /* BRISBANE_RT_SRC_STRUCTS_H */
