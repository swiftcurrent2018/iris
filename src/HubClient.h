#ifndef BRISBANE_RT_SRC_HUB_CLIENT_H
#define BRISBANE_RT_SRC_HUB_CLIENT_H

#include <sys/msg.h>

namespace brisbane {
namespace rt {

class Message;

class HubClient {
public:
    HubClient();
    ~HubClient();

    int Init();

    int IncTask(int dev, int i);
    int DecTask(int dev, int i);
    int GetNTasks(size_t* ntasks, int ndevs);

private:
    int OpenMQ();
    int CloseMQ();
    int SendMQ(Message& msg);
    int OpenPipe();
    int ClosePipe();

private:
    pid_t pid_;
    key_t key_;
    int mqid_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_HUB_CLIENT_H */
