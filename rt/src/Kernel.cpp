#include "Kernel.h"
#include "Debug.h"
#include <string.h>

namespace brisbane {
namespace rt {

Kernel::Kernel(const char* name, Platform* platform) {
    size_t len = strlen(name);
    strncpy(name_, name, len);
    name_[len] = 0;
    platform_ = platform;
    for (int i = 0; i < 16; i++) clkernels_[i] = NULL;
}

Kernel::~Kernel() {
}

int Kernel::SetArg(int idx, size_t size, void* value) {
    KernelArg* arg = new KernelArg;
    arg->size = size;
    memcpy(arg->value, value, size);
    arg->mem = platform_->GetMemFromPtr(value);
    args_[idx] = arg;
    return BRISBANE_OK;
}

cl_kernel Kernel::clkernel(int i, cl_program clprog) {
    if (clkernels_[i] == NULL) clkernels_[i] = clCreateKernel(clprog, (const char*) name_, &clerr_);
    return clkernels_[i];
}

} /* namespace rt */
} /* namespace brisbane */
