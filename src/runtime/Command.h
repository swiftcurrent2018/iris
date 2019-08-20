#ifndef BRISBANE_RT_SRC_COMMAND_H
#define BRISBANE_RT_SRC_COMMAND_H

#include <stddef.h>
#include "Kernel.h"

#define BRISBANE_CMD_NOP            0x1000
#define BRISBANE_CMD_BUILD          0x1001
#define BRISBANE_CMD_KERNEL         0x1002
#define BRISBANE_CMD_H2D            0x1003
#define BRISBANE_CMD_H2DNP          0x1004
#define BRISBANE_CMD_D2H            0x1005
#define BRISBANE_CMD_PRESENT        0x1006
#define BRISBANE_CMD_RELEASE_MEM    0x1007

namespace brisbane {
namespace rt {

class Mem;
class Task;

class Command {
public:
  Command(Task* task, int type);
  ~Command();

  int type() { return type_; }
  bool type_build() { return type_ == BRISBANE_CMD_BUILD; }
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
  Kernel* kernel() { return kernel_; }
  std::map<int, KernelArg*>* kernel_args() { return kernel_args_; }
  Mem* mem() { return mem_; }
  Task* task() { return task_; }
  bool exclusive() { return exclusive_; }
  double time() { return time_; }
  double SetTime(double t);

private:
  int type_;
  size_t size_;
  void* host_;
  int dim_;
  size_t off_[3];
  size_t ndr_[3];
  Kernel* kernel_;
  Mem* mem_;
  Task* task_;
  double time_;
  bool exclusive_;
  std::map<int, KernelArg*>* kernel_args_;

public:
  static Command* Create(Task* task, int type);
  static Command* CreateBuild(Task* task);
  static Command* CreateKernel(Task* task, Kernel* kernel, int dim, size_t* off, size_t* ndr);
  static Command* CreateH2D(Task* task, Mem* mem, size_t off, size_t size, void* host);
  static Command* CreateH2DNP(Task* task, Mem* mem, size_t off, size_t size, void* host);
  static Command* CreateD2H(Task* task, Mem* mem, size_t off, size_t size, void* host);
  static Command* CreatePresent(Task* task, Mem* mem, size_t off, size_t size, void* host);
  static Command* CreateReleaseMem(Task* task, Mem* mem);
  static Command* Duplicate(Command* cmd);
  static void Release(Command* cmd);
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_COMMAND_H */

