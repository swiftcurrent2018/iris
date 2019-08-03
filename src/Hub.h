#ifndef BRISBANE_RT_SRC_HUB_H
#define BRISBANE_RT_SRC_HUB_H

#define BRISBANE_HUB_MQ_PATH         "/tmp/brisbane_hub.mq"
#define BRISBANE_HUB_MQ_PID         52
#define BRISBANE_HUB_FIFO_PATH      "/tmp/brisbane_hub.fifo"

#define BRISBANE_HUB_MQ_MSG_SIZE    64
#define BRISBANE_HUB_MQ_REGISTER    0x1001
#define BRISBANE_HUB_MQ_DEREGISTER  0x1002
#define BRISBANE_HUB_MQ_TASK_INC    0x1003
#define BRISBANE_HUB_MQ_TASK_ALL    0x1005

#define BRISBANE_HUB_FIFO_MSG_SIZE  64
#define BRISBANE_HUB_FIFO_TASK_ALL  0x2005

#include "Config.h"
#include <sys/msg.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <map>

namespace brisbane {
namespace rt {

class Message;

class Hub {
public:
    Hub();
    ~Hub();

    int Run();

private:
    int OpenMQ();
    int CloseMQ();

    int SendFIFO(Message& msg, int pid);

    int ExecuteRegister(Message& msg, int pid);
    int ExecuteDeregister(Message& msg, int pid);
    int ExecuteTaskInc(Message& msg, int pid);
    int ExecuteTaskAll(Message& msg, int pid);

private:
    key_t key_;
    int mqid_;
    bool running_;
    std::map<int, int> fifos_;
    size_t ntasks_[BRISBANE_MAX_NDEVS];
    int ndevs_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_HUB_H */
