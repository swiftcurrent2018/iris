#ifndef BRISBANE_SRC_RT_COMMAND_H
#define BRISBANE_SRC_RT_COMMAND_H

#include <brisbane/brisbane_poly_types.h>
#include "Kernel.h"
#include <stddef.h>

#define BRISBANE_CMD_NOP            0x1000
#define BRISBANE_CMD_INIT           0x1001
#define BRISBANE_CMD_KERNEL         0x1002
#define BRISBANE_CMD_H2D            0x1003
#define BRISBANE_CMD_H2DNP          0x1004
#define BRISBANE_CMD_D2H            0x1005
#define BRISBANE_CMD_RELEASE_MEM    0x1006

namespace brisbane {
namespace rt {

class Mem;
class Task;

class Command {
public:
  Command(Task* task, int type);
  ~Command();

  int type() { return type_; }
  bool type_init() { return type_ == BRISBANE_CMD_INIT; }
  bool type_h2d() { return type_ == BRISBANE_CMD_H2D; }
  bool type_h2dnp() { return type_ == BRISBANE_CMD_H2DNP; }
  bool type_d2h() { return type_ == BRISBANE_CMD_D2H; }
  bool type_kernel() { return type_ == BRISBANE_CMD_KERNEL; }
  size_t size() { return size_; }
  void* host() { return host_; }
  int dim() { return dim_; }
  size_t* off() { return off_; }
  size_t off(int i) { return off_[i]; }
  size_t* ndr() { return ndr_; }
  size_t ndr(int i) { return ndr_[i]; }
  size_t* lws() { return lws_; }
  size_t lws(int i) { return lws_[i]; }
  Kernel* kernel() { return kernel_; }
  std::map<int, KernelArg*>* kernel_args() { return kernel_args_; }
  Mem* mem() { return mem_; }
  Task* task() { return task_; }
  bool exclusive() { return exclusive_; }
  brisbane_poly_mem* polymems() { return polymems_; }
  int npolymems() { return npolymems_; }
  double time() { return time_; }
  double SetTime(double t);

private:
  int type_;
  size_t size_;
  void* host_;
  int dim_;
  size_t off_[3];
  size_t ndr_[3];
  size_t lws_[3];
  Kernel* kernel_;
  Mem* mem_;
  Task* task_;
  double time_;
  bool exclusive_;
  std::map<int, KernelArg*>* kernel_args_;
  brisbane_poly_mem* polymems_;
  int npolymems_;

public:
  static Command* Create(Task* task, int type);
  static Command* CreateInit(Task* task);
  static Command* CreateKernel(Task* task, Kernel* kernel, int dim, size_t* off, size_t* ndr, size_t* lws);
  static Command* CreateKernelPolyMem(Task* task, Kernel* kernel, int dim, size_t* off, size_t* ndr, brisbane_poly_mem* polymems, int npolymems);
  static Command* CreateH2D(Task* task, Mem* mem, size_t off, size_t size, void* host);
  static Command* CreateH2DNP(Task* task, Mem* mem, size_t off, size_t size, void* host);
  static Command* CreateD2H(Task* task, Mem* mem, size_t off, size_t size, void* host);
  static Command* CreateReleaseMem(Task* task, Mem* mem);
  static Command* Duplicate(Command* cmd);
  static void Release(Command* cmd);
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_COMMAND_H */

