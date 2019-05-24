#include "Command.h"

namespace brisbane {
namespace rt {

Command::Command(int type) {
    type_ = type;
}

Command::~Command() {
}

Command* Command::Create(int type) {
    return new Command(type);
}

Command* Command::CreateKernel(Kernel* kernel, int dim, size_t* ndr) {
    Command* cmd = Create(BRISBANE_CMD_KERNEL);
    cmd->kernel_ = kernel;
    cmd->dim_ = dim;
    for (int i = 0; i < dim; i++) cmd->ndr_[i] = ndr[i];
    return cmd;
}

Command* Command::CreateH2D(Mem* mem, size_t off, size_t size, void* host) {
    Command* cmd = Create(BRISBANE_CMD_H2D);
    cmd->mem_ = mem;
    cmd->off_ = off;
    cmd->size_ = size;
    cmd->host_ = host;
    return cmd;
}

Command* Command::CreateD2H(Mem* mem, size_t off, size_t size, void* host) {
    Command* cmd = Create(BRISBANE_CMD_D2H);
    cmd->mem_ = mem;
    cmd->off_ = off;
    cmd->size_ = size;
    cmd->host_ = host;
    return cmd;
}

Command* Command::CreatePresent(Mem* mem, size_t off, size_t size) {
    Command* cmd = Create(BRISBANE_CMD_PRESENT);
    cmd->mem_ = mem;
    cmd->off_ = off;
    cmd->size_ = size;
    return cmd;
}

void Command::Release(Command* cmd) {
    delete cmd;
}

} /* namespace rt */
} /* namespace brisbane */

