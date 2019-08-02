#ifndef BRISBANE_RT_SRC_HUB_H
#define BRISBANE_RT_SRC_HUB_H

#define BRISBANE_HUB_MQ_KEY         "/tmp/brisbane_hub.mq"
#define BRISBANE_HUB_FIFO_PATH      "/tmp/brisbane_hub.fifo"
#define BRISBANE_HUB_MQ_PID         52

#define BRISBANE_HUB_MQ_MSG_SIZE    64

#define BRISBANE_HUB_MQ_MSG_REGISTER    0x1001
#define BRISBANE_HUB_MQ_MSG_DEREGISTER  0x1002
#define BRISBANE_HUB_MQ_MSG_INC_TASK    0x1003
#define BRISBANE_HUB_MQ_MSG_DEC_TASK    0x1004
#define BRISBANE_HUB_MQ_MSG_NTASKS      0x1005

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

    int ExecuteRegister(Message& msg, int pid);
    int ExecuteDeregister(Message& msg, int pid);

private:
    key_t key_;
    int mqid_;
    bool running_;
    std::map<int, int> fifos_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_HUB_H */
