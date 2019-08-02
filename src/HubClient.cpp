#include <brisbane/brisbane.h>
#include "HubClient.h"
#include "Hub.h"
#include "Debug.h"
#include "Message.h"
#include <unistd.h>

namespace brisbane {
namespace rt {

HubClient::HubClient() {
    pid_ = getpid();
}

HubClient::~HubClient() {
    CloseMQ();
}

int HubClient::Init() {
    _check();
    int ret = OpenMQ();
    if (ret == BRISBANE_OK) ret = OpenPipe();
    return ret;
}

int HubClient::OpenMQ() {
    _check();
    if ((key_ = ftok(BRISBANE_HUB_MQ_KEY, BRISBANE_HUB_MQ_PID)) == -1) return BRISBANE_ERR;
    if ((mqid_ = msgget(key_, 0644 | IPC_CREAT)) == -1) return BRISBANE_ERR;
    return BRISBANE_OK;
}

int HubClient::CloseMQ() {
    ClosePipe();
    return BRISBANE_OK;
}

int HubClient::SendMQ(Message& msg) {
    int iret = msgsnd(mqid_, msg.buf(), BRISBANE_HUB_MQ_MSG_SIZE, 0);
    if (iret == -1) {
        _error("msgsnd err[%d]", iret);
        perror("msgsnd");
        return BRISBANE_ERR;
    }
    return BRISBANE_OK;
}

int HubClient::OpenPipe() {
    Message msg(BRISBANE_HUB_MQ_MSG_REGISTER);
    msg.WriteInt(pid_);
    SendMQ(msg);
    return BRISBANE_OK;
}

int HubClient::ClosePipe() {
    _check();
    Message msg(BRISBANE_HUB_MQ_MSG_DEREGISTER);
    msg.WriteInt(pid_);
    SendMQ(msg);
    return BRISBANE_OK;
}

int HubClient::IncTask(int dev, int i) {

    return BRISBANE_OK;
}

int HubClient::DecTask(int dev, int i) {

    return BRISBANE_OK;
}

int HubClient::GetNTasks(size_t* ntasks, int ndevs) {

    return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */
