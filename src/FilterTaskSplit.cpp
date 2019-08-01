#include "FilterTaskSplit.h"
#include "Config.h"
#include "Debug.h"
#include "Command.h"
#include "Kernel.h"
#include "Mem.h"
#include "Polyhedral.h"
#include "Task.h"

namespace brisbane {
namespace rt {

FilterTaskSplit::FilterTaskSplit(Polyhedral* polyhedral, Platform* platform) {
    polyhedral_ = polyhedral;
    platform_ = platform;
}

FilterTaskSplit::~FilterTaskSplit() {
}

int FilterTaskSplit::Execute(Task* task) {
    Command* cmd_kernel = task->cmd_kernel();
    if (!cmd_kernel) return BRISBANE_OK;
    Kernel* kernel = cmd_kernel->kernel();

    polyhedral_->Kernel(kernel->name());
    int dim = cmd_kernel->dim();
    size_t off[3];
    size_t ndr[3];
    for (int i = 0; i < 3; i++) {
        off[i] = cmd_kernel->off(i);
        ndr[i] = cmd_kernel->ndr(i);
    }
    int nmems = 0;
    std::map<int, KernelArg*>* args = cmd_kernel->kernel_args();
    for (std::map<int, KernelArg*>::iterator it = args->begin(); it != args->end(); ++it) {
        int idx = it->first;
        KernelArg* arg = it->second;
        Mem* mem = arg->mem;
        if (mem) nmems++;
        else polyhedral_->SetArg(idx, arg->size, arg->value); 
    }

    brisbane_poly_mem* plmems = new brisbane_poly_mem[nmems];
    Mem* plmems_mem[nmems];
    size_t chunk_size = 4;
    size_t ndr0 = ndr[0];
    bool left_ndr = ndr[0] % chunk_size;
    size_t nchunks = ndr[0] / chunk_size + (left_ndr ? 1 : 0);
    Task** subtasks = new Task*[nchunks];
    for (size_t i = 0; i < nchunks; i++) {
        subtasks[i] = new Task(platform_);
        off[0] = i * chunk_size;
        if (left_ndr && i == nchunks - 1) ndr[0] = ndr0 - i * chunk_size;
        else ndr[0] = chunk_size;
        polyhedral_->Launch(dim, off, ndr);
        int mem_idx = 0;
        for (std::map<int, KernelArg*>::iterator it = args->begin(); it != args->end(); ++it) {
            int idx = it->first;
            KernelArg* arg = it->second;
            Mem* mem = arg->mem;
            if (mem) {
                polyhedral_->GetMem(idx, plmems + mem_idx);
                _debug("idx[%d] mem[%lu] typesz[%lu] off[%lu,%lu] len[%lu,%lu]", idx, mem->uid(), plmems[mem_idx].typesz, plmems[mem_idx].off_r, plmems[mem_idx].off_w, plmems[mem_idx].len_r, plmems[mem_idx].len_w);
                plmems_mem[mem_idx] = mem;
                mem_idx++;
            }
        }
        for (int j = 0; j < task->ncmds(); j++) {
            Command* cmd = task->cmd(j);
            if (cmd->type_h2d()) {
                Mem* mem = cmd->mem();
                for (int k = 0; k < nmems; k++) {
                    if (plmems_mem[k] == mem) {
                        brisbane_poly_mem* plmem = plmems + k; 
                        Command* sub_cmd = Command::CreateH2D(subtasks[i], mem, plmem->typesz * plmem->off_r, plmem->typesz * plmem->len_r, (char*) cmd->host() + plmem->typesz * plmem->off_r);
                        subtasks[i]->AddCommand(sub_cmd);
                    }
                }
            } else if (cmd->type_d2h()) {
                Mem* mem = cmd->mem();
                for (int k = 0; k < nmems; k++) {
                    if (plmems_mem[k] == mem) {
                        brisbane_poly_mem* plmem = plmems + k; 
                        Command* sub_cmd = Command::CreateD2H(subtasks[i], mem, plmem->typesz * plmem->off_w, plmem->typesz * plmem->len_w, (char*) cmd->host() + plmem->typesz * plmem->off_w);
                        subtasks[i]->AddCommand(sub_cmd);
                    }
                }

            } else if (cmd->type_kernel()) {
                Kernel* kernel = cmd->kernel();
                Command* sub_cmd = Command::CreateKernel(subtasks[i], kernel, cmd->dim(), off, ndr);
                subtasks[i]->AddCommand(sub_cmd);
            }
        }
        task->AddSubtask(subtasks[i]);
    }
    task->ClearCommands();
    delete[] plmems;

    return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */
