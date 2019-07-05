#include "Kernel.h"
#include "Debug.h"
#include "History.h"
#include <string.h>

namespace brisbane {
namespace rt {

Kernel::Kernel(const char* name, Platform* platform) {
    size_t len = strlen(name);
    strncpy(name_, name, len);
    name_[len] = 0;
    platform_ = platform;
    history_ = new History(this);
    for (int i = 0; i < BRISBANE_MAX_NDEVS; i++) clkernels_[i] = NULL;
}

Kernel::~Kernel() {
    for (std::map<int, KernelArg*>::iterator it = args_.begin(); it != args_.end(); ++it)
        delete it->second;
}

int Kernel::SetArg(int idx, size_t size, void* value) {
    KernelArg* arg = new KernelArg;
    arg->size = size;
    if (value) memcpy(arg->value, value, size);
    arg->mem = NULL;
    args_[idx] = arg;
    return BRISBANE_OK;
}

int Kernel::SetMem(int idx, Mem* mem, int mode) {
    KernelArg* arg = new KernelArg;
    arg->mem = mem;
    arg->mode = mode;
    args_[idx] = arg;
    return BRISBANE_OK;
}

std::map<int, KernelArg*>* Kernel::ExportArgs() {
    std::map<int, KernelArg*>* new_args = new std::map<int, KernelArg*>();
    for (std::map<int, KernelArg*>::iterator it = args_.begin(); it != args_.end(); ++it) {
        KernelArg* arg = it->second;
        KernelArg* new_arg = new KernelArg;
        if (arg->mem) {
            new_arg->mem = arg->mem;
            new_arg->mode = arg->mode;
        } else {
            new_arg->size = arg->size; 
            memcpy(new_arg->value, arg->value, arg->size);
            new_arg->mem = NULL;
        }
        (*new_args)[it->first] = new_arg;
    }
    return new_args;
}

cl_kernel Kernel::clkernel(int i, cl_program clprog) {
    if (clkernels_[i] == NULL) {
        clkernels_[i] = clCreateKernel(clprog, (const char*) name_, &clerr_);
        _clerror(clerr_);
    }
    return clkernels_[i];
}

} /* namespace rt */
} /* namespace brisbane */
