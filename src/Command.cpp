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

Command* Command::CreateKernel(Kernel* kernel, int dim, size_t* off, size_t* ndr) {
    Command* cmd = Create(BRISBANE_CMD_KERNEL);
    cmd->kernel_ = kernel;
    cmd->dim_ = dim;
    for (int i = 0; i < dim; i++) {
        cmd->off_[i] = off[i];
        cmd->ndr_[i] = ndr[i];
    }
    for (int i = dim; i < 3; i++) {
        cmd->off_[i] = 0;
        cmd->ndr_[i] = 1;
    }
    return cmd;
}

Command* Command::CreateH2D(Mem* mem, size_t off, size_t size, void* host) {
    Command* cmd = Create(BRISBANE_CMD_H2D);
    cmd->mem_ = mem;
    cmd->off_[0] = off;
    cmd->size_ = size;
    cmd->host_ = host;
    return cmd;
}

Command* Command::CreateD2H(Mem* mem, size_t off, size_t size, void* host) {
    Command* cmd = Create(BRISBANE_CMD_D2H);
    cmd->mem_ = mem;
    cmd->off_[0] = off;
    cmd->size_ = size;
    cmd->host_ = host;
    return cmd;
}

Command* Command::CreatePresent(Mem* mem, size_t off, size_t size, void* host) {
    Command* cmd = Create(BRISBANE_CMD_PRESENT);
    cmd->mem_ = mem;
    cmd->off_[0] = off;
    cmd->size_ = size;
    cmd->host_ = host;
    return cmd;
}

void Command::Release(Command* cmd) {
    delete cmd;
}

} /* namespace rt */
} /* namespace brisbane */

