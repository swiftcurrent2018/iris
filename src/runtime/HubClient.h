#ifndef BRISBANE_SRC_RT_HUB_CLIENT_H
#define BRISBANE_SRC_RT_HUB_CLIENT_H

#include <sys/msg.h>

namespace brisbane {
namespace rt {

class Message;
class Scheduler;

class HubClient {
public:
  HubClient(Scheduler* scheduler);
  ~HubClient();

  int Init();
  int StopHub();
  int Status();

  int TaskInc(int dev, int i);
  int TaskDec(int dev, int i);
  int TaskAll(size_t* ntasks, int ndevs);

  bool available() { return available_; }

private:
  int OpenMQ();
  int CloseMQ();
  int SendMQ(Message& msg);

  int OpenFIFO();
  int CloseFIFO();
  int RecvFIFO(Message& msg);

  int Register();
  int Deregister();

private:
  Scheduler* scheduler_;
  pid_t pid_;
  key_t key_;
  int mq_;
  int fifo_;
  int ndevs_;
  bool available_;
  bool stop_hub_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_HUB_CLIENT_H */

