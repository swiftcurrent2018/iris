#include "Charts.h"
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

Charts::Charts(Platform* platform) {
  platform_ = platform;
  first_task_ = 0;
  fd_ = -1;
  OpenFD();
  Main();
}

Charts::~Charts() {
  Exit();
  CloseFD();
}

int Charts::AddTask(Task* task) {
  if (first_task_ == 0) first_task_ = (unsigned long) (task->time_start() * 1.e+6);

  unsigned long tid = task->uid();
  Device* dev = task->dev();
  char s[256];
  sprintf(s, "[ '%s', '%s', %lu, %lu ],", dev->name(), task->name(),
      (unsigned long) (task->time_start() * 1.e+6) - first_task_,
      (unsigned long) (task->time_end() * 1.e+6) - first_task_);
  Write(s);
  return BRISBANE_OK;
}

int Charts::OpenFD() {
  time_t t = time(NULL);
  char s[64];
  strftime(s, 64, "%Y%m%d-%H%M%S", localtime(&t));
  sprintf(path_, "%s-%s-%s.html", platform_->app(), platform_->host(), s);
  _debug("dot path[%s]", path_);
  fd_ = open(path_, O_CREAT | O_WRONLY, 0666);
  if (fd_ == -1) {
    _error("open chart file[%s]", path_);
    perror("open");
    return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

int Charts::CloseFD() {
  if (fd_ != -1) {
    int iret = close(fd_);
    if (iret == -1) {
      _error("close chart file[%s]", path_);
      perror("close");
    }
  }
}

int Charts::Main() {
  Write((char*) "<script type='text/javascript' src='https://www.gstatic.com/charts/loader.js'></script>");
  Write((char*) "<script type='text/javascript'>");
  Write((char*) "google.charts.load('current', {packages:['timeline']});");
  Write((char*) "google.charts.setOnLoadCallback(drawChart);");
  Write((char*) "function drawChart() {");
  Write((char*) "var container = document.getElementById('brisbane');");
  Write((char*) "var chart = new google.visualization.Timeline(container);");
  Write((char*) "var dataTable = new google.visualization.DataTable();");
  Write((char*) "dataTable.addColumn({ type: 'string', id: 'Device' });");
  Write((char*) "dataTable.addColumn({ type: 'string', id: 'Task' });");
  Write((char*) "dataTable.addColumn({ type: 'number', id: 'Start' });");
  Write((char*) "dataTable.addColumn({ type: 'number', id: 'End' });");
  Write((char*) "dataTable.addRows([");
  return BRISBANE_OK;
}

int Charts::Exit() {
  Write((char*) "]);");
  Write((char*) "var options = {");
  Write((char*) "timeline: { colorByRowLabel: true }");
  Write((char*) "};");
  Write((char*) "chart.draw(dataTable, options);");
  Write((char*) "}");
  Write((char*) "</script>");
  Write((char*) "<div id='brisbane' style='height: 100%;'></div>");
  return BRISBANE_OK;
}

int Charts::Write(char* s, bool tab) {
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

} /* namespace rt */
} /* namespace brisbane */

