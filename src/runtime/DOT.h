#ifndef BRISBANE_RT_SRC_DOT_H
#define BRISBANE_RT_SRC_DOT_H

#include <set>

namespace brisbane {
namespace rt {

class Platform;
class Task;

class DOT {
public:
  DOT(Platform* platform);
  ~DOT();

  int AddTask(Task* task);

private:
  int OpenFD();
  int CloseFD();
  int Main();
  int Exit();
  int Write(char* s, bool tab = true);

private:
  Platform* platform_;
  int fd_;
  char path_[256];
  std::set<unsigned long> tasks_exit_;
};

} /* namespace rt */
} /* namespace brisbane */


#endif /*BRISBANE_RT_SRC_DOT_H */
