#include <brisbane/brisbane.h>
#include "HubClient.h"
#include "Hub.h"
#include "Debug.h"
#include "Message.h"
#include <stdlib.h>
#include <unistd.h>

namespace brisbane {
namespace rt {

HubClient::HubClient() {
    pid_ = getpid();
    fifo_ = -1;
}

HubClient::~HubClient() {
    CloseMQ();
}

int HubClient::Init() {
    int ret = OpenMQ();
    if (ret == BRISBANE_OK) ret = OpenFIFO();
    return ret;
}

int HubClient::OpenMQ() {
    if ((key_ = ftok(BRISBANE_HUB_MQ_PATH, BRISBANE_HUB_MQ_PID)) == -1) return BRISBANE_ERR;
    if ((mq_ = msgget(key_, 0644 | IPC_CREAT)) == -1) return BRISBANE_ERR;
    return BRISBANE_OK;
}

int HubClient::CloseMQ() {
    if (fifo_ != -1) CloseFIFO();
    return BRISBANE_OK;
}

int HubClient::SendMQ(Message& msg) {
    int iret = msgsnd(mq_, msg.buf(), BRISBANE_HUB_MQ_MSG_SIZE, 0);
    if (iret == -1) {
        _error("msgsnd err[%d]", iret);
        perror("msgsnd");
        return BRISBANE_ERR;
    }
    return BRISBANE_OK;
}

int HubClient::OpenFIFO() {
    char path[64];
    sprintf(path, "%s.%d", BRISBANE_HUB_FIFO_PATH, pid_);
    int iret = mknod(path, S_IFIFO | 0640, 0);
    if (iret == -1) {
        _error("iret[%d]", iret);
        perror("mknod");
        return BRISBANE_ERR;
    }
    fifo_ = open(path, O_RDWR);
    if (fifo_ == -1) {
        _error("path[%s]", path);
        perror("read");
        return BRISBANE_ERR;
    }
    _debug("open fifo[%s]", path);
    Message msg(BRISBANE_HUB_MQ_REGISTER);
    msg.WritePID(pid_);
    SendMQ(msg);
    return BRISBANE_OK;
}

int HubClient::CloseFIFO() {
    Message msg(BRISBANE_HUB_MQ_DEREGISTER);
    msg.WritePID(pid_);
    SendMQ(msg);
    return BRISBANE_OK;
}

int HubClient::RecvFIFO(Message& msg) {
    ssize_t ssret = read(fifo_, msg.buf(), BRISBANE_HUB_FIFO_MSG_SIZE);
    if (ssret != BRISBANE_HUB_FIFO_MSG_SIZE) {
        _error("ssret[%ld]", ssret);
        perror("read");
        return BRISBANE_ERR;
    }
    return BRISBANE_OK;
}

int HubClient::TaskInc(int dev, int i) {
    _debug("dev[%d] i[%d]", dev, i);
    Message msg(BRISBANE_HUB_MQ_TASK_INC);
    msg.WritePID(pid_);
    msg.WriteInt(dev);
    msg.WriteInt(i);
    SendMQ(msg);
    return BRISBANE_OK;
}

int HubClient::TaskDec(int dev, int i) {
    return TaskInc(dev, -i);
}

int HubClient::TaskAll(size_t* ntasks, int ndevs) {
    _debug("ndevs[%d]", ndevs);
    Message msg(BRISBANE_HUB_MQ_TASK_ALL);
    msg.WritePID(pid_);
    msg.WriteInt(ndevs);
    SendMQ(msg);

    msg.Clear();
    RecvFIFO(msg);
    int header = msg.ReadHeader();
    if (header != BRISBANE_HUB_FIFO_TASK_ALL) {
        _error("header[0x%x]", header);
    }
    for (int i = 0; i < ndevs; i++) {
        ntasks[i] = msg.ReadULong();
        _debug("dev[%d] ntasks[%lu]", i, ntasks[i]);
    }
    return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */
