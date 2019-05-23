#ifndef BRISBANE_RT_SRC_KERNEL_H
#define BRISBANE_RT_SRC_KERNEL_H

#include "Object.h"
#include "Platform.h"
#include <map>

namespace brisbane {
namespace rt {

typedef struct _KernelArg {
    size_t size;
    char value[256];
    Mem* mem;
} KernelArg;

class Kernel: public Object<struct _brisbane_kernel, Kernel> {
public:
    Kernel(const char* name, Platform* platform);
    virtual ~Kernel();

    int SetArg(int idx, size_t size, void* value);

    cl_kernel clkernel(int i, cl_program clprog);

    std::map<int, KernelArg*> args() { return args_; }
    char* name() { return name_; }

private:
    char name_[256];
    std::map<int, KernelArg*> args_;
    cl_kernel clkernels_[16];

    Platform* platform_;

    cl_int clerr_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_KERNEL_H */
