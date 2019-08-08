#ifndef BRISBANE_RT_SRC_CHARTS_H
#define BRISBANE_RT_SRC_CHARTS_H

namespace brisbane {
namespace rt {

class Platform;
class Task;

class Charts {
public:
  Charts(Platform* platform);
  ~Charts();

  int AddTask(Task* task);

private:
  int OpenFD();
  int CloseFD();
  int Main();
  int Exit();
  int Write(char* s, bool tab = false);

private:
  Platform* platform_;
  int fd_;
  char path_[256];
  unsigned long first_task_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /*BRISBANE_RT_SRC_CHARTS_H */

