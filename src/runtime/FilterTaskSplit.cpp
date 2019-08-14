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
  int nmems = 0;
  std::map<int, KernelArg*>* args = cmd_kernel->kernel_args();
  for (std::map<int, KernelArg*>::iterator I = args->begin(), E = args->end(); I != E; ++I) {
    int idx = I->first;
    KernelArg* arg = I->second;
    Mem* mem = arg->mem;
    if (mem) nmems++;
    else polyhedral_->SetArg(idx, arg->size, arg->value); 
  }

  int dim = cmd_kernel->dim();
  size_t off[3];
  size_t ndr[3];
  for (int i = 0; i < 3; i++) {
    off[i] = cmd_kernel->off(i);
    ndr[i] = cmd_kernel->ndr(i);
  }

  size_t wgo[3] = { off[0], off[1], off[2] };
  size_t wgs[3] = { ndr[0], ndr[1], ndr[2] };
  size_t gws[3] = { ndr[0], ndr[1], ndr[2] };
  size_t lws[3] = { 1, 1, 1 };

  brisbane_poly_mem* plmems = new brisbane_poly_mem[nmems];
  Mem* plmems_mem[nmems];
  size_t chunk_size = ndr[0] / (platform_->ndevs() * 4);
  size_t ndr0 = ndr[0];
  bool left_ndr = ndr[0] % chunk_size;
  size_t nchunks = ndr[0] / chunk_size + (left_ndr ? 1 : 0);
  Task** subtasks = new Task*[nchunks];
  for (size_t i = 0; i < nchunks; i++) {
    subtasks[i] = new Task(platform_);
    off[0] = i * chunk_size;
    if (left_ndr && i == nchunks - 1) ndr[0] = ndr0 - i * chunk_size;
    else ndr[0] = chunk_size;

    wgo[0] = off[0];
    wgs[0] = ndr[0];

    polyhedral_->Launch(dim, wgo, wgs, gws, lws);
    int mem_idx = 0;
    for (std::map<int, KernelArg*>::iterator I = args->begin(), E = args->end(); I != E; ++I) {
      int idx = I->first;
      KernelArg* arg = I->second;
      Mem* mem = arg->mem;
      if (mem) {
        polyhedral_->GetMem(idx, plmems + mem_idx);
        _debug("idx[%d] mem[%lu] typesz[%lu] read[%lu,%lu] write[%lu,%lu]", idx, mem->uid(), plmems[mem_idx].typesz, plmems[mem_idx].r0, plmems[mem_idx].r1, plmems[mem_idx].w0, plmems[mem_idx].w1);
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
            Command* sub_cmd = Command::CreateH2D(subtasks[i], mem, plmem->typesz * plmem->r0, plmem->typesz * (plmem->r1 - plmem->r0 + 1), (char*) cmd->host() + plmem->typesz * plmem->r0);
            subtasks[i]->AddCommand(sub_cmd);
          }
        }
      } else if (cmd->type_d2h()) {
        Mem* mem = cmd->mem();
        for (int k = 0; k < nmems; k++) {
          if (plmems_mem[k] == mem) {
            brisbane_poly_mem* plmem = plmems + k; 
            Command* sub_cmd = Command::CreateD2H(subtasks[i], mem, plmem->typesz * plmem->w0, plmem->typesz * (plmem->w1 - plmem->w0 + 1), (char*) cmd->host() + plmem->typesz * plmem->w0);
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
