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
}

Hub::~Hub() {
    _check();
    CloseMQ();
}

int Hub::OpenMQ() {
    _check();
    char cmd[64];
    memset(cmd, 0, 64);
    sprintf(cmd, "touch %s", BRISBANE_HUB_MQ_KEY);
    if (system(cmd) == -1) perror(cmd);
    if ((key_ = ftok(BRISBANE_HUB_MQ_KEY, BRISBANE_HUB_MQ_PID)) == -1) {
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
    sprintf(cmd, "rm -f %s", BRISBANE_HUB_MQ_KEY);
    system(cmd);
    return BRISBANE_OK;
}

int Hub::Run() {
    _check();
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
        int pid = msg.ReadInt();
        switch (header) {
            _debug("header[0x%x]", header);
            case BRISBANE_HUB_MQ_MSG_REGISTER:      ret = ExecuteRegister(msg, pid);    break;
            case BRISBANE_HUB_MQ_MSG_DEREGISTER:    ret = ExecuteDeregister(msg, pid);  break;
            default: _error("not supported msg header[0x%x", header);
        }
        if (ret != BRISBANE_OK) _error("header[0x%x] ret[%d]", header, ret);
    }
    return BRISBANE_OK;
}

int Hub::ExecuteRegister(Message& msg, int pid) {
    _debug("pid[%d]", pid);
    char path[64];
    sprintf(path, "%s.%d", BRISBANE_HUB_FIFO_PATH, pid);
    int iret = mknod(path, S_IFIFO | 0640, 0);
    if (iret == -1) {
        _error("iret[%d]", iret);
        perror("mknod");
        return BRISBANE_ERR;
    }
    int fd = open(path, O_RDWR);
    fifos_[pid] = fd;
    return BRISBANE_OK;
}

int Hub::ExecuteDeregister(Message& msg, int pid) {
    _debug("pid[%d]", pid);
    int fifo = fifos_[pid];
    int iret = close(fifo);
    if (iret == -1) {
        _error("iret[%d]", iret);
        perror("close");
        return BRISBANE_ERR;
    }
    fifos_.erase(pid);
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
