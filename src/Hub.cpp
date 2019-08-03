#include <brisbane/brisbane.h>
#include "Hub.h"
#include "Debug.h"
#include "Message.h"
#include <stdlib.h>
#include <signal.h>

namespace brisbane {
namespace rt {

char brisbane_log_prefix_[256];

Hub::Hub() {
    sprintf(brisbane_log_prefix_, "BRISBANE_HUB");
    running_ = false;
    mqid_ = -1;
    int ndevs_ = BRISBANE_MAX_NDEVS;
    for (int i = 0; i < ndevs_; i++) ntasks_[i] = 0;
}

Hub::~Hub() {
    _check();
    CloseMQ();
}

int Hub::OpenMQ() {
    _check();
    char cmd[64];
    memset(cmd, 0, 64);
    sprintf(cmd, "touch %s", BRISBANE_HUB_MQ_PATH);
    if (system(cmd) == -1) perror(cmd);
    if ((key_ = ftok(BRISBANE_HUB_MQ_PATH, BRISBANE_HUB_MQ_PID)) == -1) {
        perror("ftok");
        return BRISBANE_ERR;
    }
    if ((mqid_ = msgget(key_, 0644 | IPC_CREAT)) == -1) {
        perror("msgget");
        return BRISBANE_ERR;
    }
    return BRISBANE_OK;
}

int Hub::CloseMQ() {
    _check();
    char cmd[64];
    memset(cmd, 0, 64);
    sprintf(cmd, "rm -f %s %s*", BRISBANE_HUB_MQ_PATH, BRISBANE_HUB_FIFO_PATH);
    system(cmd);
    return BRISBANE_OK;
}

int Hub::SendFIFO(Message& msg, int pid) {
    int fd = fifos_[pid];
    ssize_t ssret = write(fd, msg.buf(), BRISBANE_HUB_FIFO_MSG_SIZE);
    if (ssret != BRISBANE_HUB_FIFO_MSG_SIZE) {
        _error("ssret[%ld]", ssret);
        perror("write");
    }
    return BRISBANE_OK;
}

int Hub::Run() {
    OpenMQ();
    running_ = true;
    Message msg;
    int ret;
    while (running_) {
        msg.Clear();
        if (msgrcv(mqid_, msg.buf(), BRISBANE_HUB_MQ_MSG_SIZE, 0, 0) == -1) {
            perror("msgrcv");
            continue;
        }
        int header = msg.ReadHeader();
        int pid = msg.ReadPID();
        switch (header) {
            _debug("header[0x%x]", header);
            case BRISBANE_HUB_MQ_REGISTER:      ret = ExecuteRegister(msg, pid);    break;
            case BRISBANE_HUB_MQ_DEREGISTER:    ret = ExecuteDeregister(msg, pid);  break;
            case BRISBANE_HUB_MQ_TASK_INC:      ret = ExecuteTaskInc(msg, pid);     break;
            case BRISBANE_HUB_MQ_TASK_ALL:      ret = ExecuteTaskAll(msg, pid);     break;
            default: _error("not supported msg header[0x%x", header);
        }
        if (ret != BRISBANE_OK) _error("header[0x%x] ret[%d]", header, ret);
    }
    return BRISBANE_OK;
}

int Hub::ExecuteRegister(Message& msg, int pid) {
    char path[64];
    sprintf(path, "%s.%d", BRISBANE_HUB_FIFO_PATH, pid);
    _debug("open fifo[%s]", path);
    int fd = open(path, O_RDWR);
    fifos_[pid] = fd;
    return BRISBANE_OK;
}

int Hub::ExecuteDeregister(Message& msg, int pid) {
    char path[64];
    sprintf(path, "%s.%d", BRISBANE_HUB_FIFO_PATH, pid);
    _debug("pid[%d]", pid);
    int fifo = fifos_[pid];
    int iret = close(fifo);
    if (iret == -1) {
        _error("iret[%d]", iret);
        perror("close");
    }
    iret = remove(path);
    if (iret == -1) {
        _error("iret[%d]", iret);
        perror("remove");
    }
    fifos_.erase(pid);
    return BRISBANE_OK;
}

int Hub::ExecuteTaskInc(Message& msg, int pid) {
    int dev = msg.ReadInt();
    int i = msg.ReadInt();
    ntasks_[dev] += i;
    _debug("dev[%d] i[%d] ntasks[%lu]", dev, i, ntasks_[dev]);
    return BRISBANE_OK;
}

int Hub::ExecuteTaskAll(Message& msg, int pid) {
    int ndevs = msg.ReadInt();
    _debug("ndevs[%d]", ndevs);
    for (int i = 0; i < ndevs; i++) {
        _debug("dev[%d] ntasks[%lu]", i, ntasks_[i]);
    }
    Message fmsg(BRISBANE_HUB_FIFO_TASK_ALL);
    fmsg.Write(ntasks_, ndevs * sizeof(size_t));
    SendFIFO(fmsg, pid);
    return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */

using namespace brisbane::rt;

Hub* hub;

void brisbane_hub_sigint_handler(int signum) {
    if (signum == SIGINT) {
        if (hub) delete hub;
        exit(0);
    } else {
        _debug("signum[%d]", signum);
    }
}

int main(int argc, char** argv) {
    hub = new Hub();
    if (argc > 1 && strcmp("kill", argv[1]) == 0) {
        delete hub;
        return 0;
    }
    signal(SIGINT, brisbane_hub_sigint_handler);
    hub->Run();
    delete hub;
    return 0;
}
