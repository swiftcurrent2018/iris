#include "DOT.h"
#include "Debug.h"
#include "Device.h"
#include "Platform.h"
#include "Task.h"
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

namespace brisbane {
namespace rt {

DOT::DOT(Platform* platform) {
  platform_ = platform;
  fd_ = -1;
  OpenFD();
  Main();
}

DOT::~DOT() {
  Exit();
  CloseFD();
}

int DOT::OpenFD() {
  time_t t = time(NULL);
  char s[64];
  strftime(s, 64, "%Y%m%d-%H%M%S", localtime(&t));
  sprintf(path_, "%s-%s-%s.dot", platform_->app(), platform_->host(), s);
  _debug("dot path[%s]", path_);
  fd_ = open(path_, O_CREAT | O_WRONLY, 0666);
  if (fd_ == -1) {
    _error("open dot file[%s]", path_);
    perror("open");
    return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

int DOT::Main() {
  Write((char*) "digraph {", false);
  char s[256];
  sprintf(s, "start[shape=Mdiamond, label=\"%s\"]", "main");
  Write(s);
  return BRISBANE_OK;
}

int DOT::Exit() {
  char s[256];
  for (std::set<unsigned long>::iterator I = tasks_exit_.begin(), E = tasks_exit_.end(); I != E; ++I) {
    unsigned long tid = *I;
    sprintf(s, "task%lu -> end", tid);
    Write(s);
  }
  sprintf(s, "end[shape=Msquare, label=\"exit\\n%lf\"]", platform_->app(), platform_->time_app());
  Write(s);
  Write((char*) "}", false);
  return BRISBANE_OK;
}

int DOT::AddTask(Task* task) {
  unsigned long tid = task->uid();
  Device* dev = task->dev();
  int type = dev->type();
  int policy = task->brs_device();
  double time = task->time();
  tasks_exit_.insert(tid);
  char s[256];
  sprintf(s, "task%lu[style=filled, fillcolor=%s, label=\"%s (%s)\\n%lf\"]", tid,
      type & brisbane_cpu ? "cyan" : type & brisbane_gpu ? "green" : "purple", task->name(),
      policy == brisbane_default  ? "default" :
      policy == brisbane_cpu      ? "cpu" :
      policy == brisbane_nvidia   ? "nvidia" :
      policy == brisbane_amd      ? "amd" :
      policy == brisbane_gpu      ? "gpu" :
      policy == brisbane_phi      ? "phi" :
      policy == brisbane_fpga     ? "fpga" :
      policy == brisbane_data     ? "data" :
      policy == brisbane_profile  ? "profile" :
      policy == brisbane_eager    ? "eager" :
      policy == brisbane_random   ? "random" :
      policy == brisbane_any      ? "any" :
      policy &  brisbane_all      ? "all" : "?", time);
  Write(s);

  int ndepends = task->ndepends();
  if (ndepends == 0) {
    sprintf(s, "start -> task%lu", tid);
    Write(s);
  } else {
    Task** deps = task->depends();
    for (int i = 0; i < ndepends; i++) {
      unsigned long duid = deps[i]->uid();
      tasks_exit_.erase(duid);
      sprintf(s, "task%lu -> task%lu", duid, tid);
      Write(s);
    }
  }
  return BRISBANE_OK;
}

int DOT::Write(char* s, bool tab) {
  char tabs[256];
  sprintf(tabs, tab ? "  %s\n" : "%s\n", s);
  size_t len = strlen(tabs);
  ssize_t ssret = write(fd_, tabs, len);
  if (ssret != len) {
    _error("path[%s] str[%s] ssret[%ld]", path_, tabs, ssret);
    perror("write");
    return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

int DOT::CloseFD() {
  if (fd_ != -1) {
    int iret = close(fd_);
    if (iret == -1) {
      _error("close dot file[%s]", path_);
      perror("close");
    }
  }
}


} /* namespace rt */
} /* namespace brisbane */
