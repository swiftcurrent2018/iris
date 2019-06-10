#ifndef BRISBANE_RT_SRC_COMMAND_H
#define BRISBANE_RT_SRC_COMMAND_H

#include <stddef.h>

#define BRISBANE_CMD_NOP        0x1000
#define BRISBANE_CMD_KERNEL     0x1001
#define BRISBANE_CMD_H2D        0x1002
#define BRISBANE_CMD_D2H        0x1003
#define BRISBANE_CMD_PRESENT    0x1004

namespace brisbane {
namespace rt {

class Kernel;
class Mem;

class Command {
public:
    Command(int type);
    ~Command();

    int type() { return type_; }
    size_t size() { return size_; }
    void* host() { return host_; }
    int dim() { return dim_; }
    size_t* off() { return off_; }
    size_t off(int i) { return off_[i]; }
    size_t* ndr() { return ndr_; }
    size_t ndr(int i) { return ndr_[i]; }
    Kernel* kernel() { return kernel_; }
    Mem* mem() { return mem_; }

private:
    int type_;
    size_t size_;
    void* host_;
    int dim_;
    size_t off_[3];
    size_t ndr_[3];
    Kernel* kernel_;
    Mem* mem_;

public:
    static Command* Create(int type);
    static Command* CreateKernel(Kernel* kernel, int dim, size_t* off, size_t* ndr);
    static Command* CreateH2D(Mem* mem, size_t off, size_t size, void* host);
    static Command* CreateD2H(Mem* mem, size_t off, size_t size, void* host);
    static Command* CreatePresent(Mem* mem, size_t off, size_t size, void* host);
    static void Release(Command* cmd);
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_COMMAND_H */
