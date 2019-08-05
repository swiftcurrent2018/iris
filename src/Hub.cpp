#include <brisbane/brisbane.h>
#include "Hub.h"
#include "HubClient.h"
#include "Debug.h"
#include "Message.h"
#include <stdlib.h>

namespace brisbane {
namespace rt {

char brisbane_log_prefix_[32] = "BRISBANE-HUB";
const static char* app = "BRISBANE-HUB";

Hub::Hub() {
    running_ = false;
    mq_ = -1;
    ndevs_ = 0;
    for (int i = 0; i < BRISBANE_MAX_NDEVS; i++) ntasks_[i] = 0;
}

Hub::~Hub() {
    if (mq_ != -1) CloseMQ();
}

int Hub::OpenMQ() {
    char cmd[64];
    memset(cmd, 0, 64);
    sprintf(cmd, "touch %s", BRISBANE_HUB_MQ_PATH);
    if (system(cmd) == -1) perror(cmd);
    if ((key_ = ftok(BRISBANE_HUB_MQ_PATH, BRISBANE_HUB_MQ_PID)) == -1) {
        perror("ftok");
        return BRISBANE_ERR;
    }
    if ((mq_ = msgget(key_, 0644 | IPC_CREAT)) == -1) {
        perror("msgget");
        return BRISBANE_ERR;
    }
    return BRISBANE_OK;
}

int Hub::CloseMQ() {
    char cmd[64];
    memset(cmd, 0, 64);
    sprintf(cmd, "rm -f %s %s*", BRISBANE_HUB_MQ_PATH, BRISBANE_HUB_FIFO_PATH);
    if (system(cmd) == -1) perror(cmd);
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
    while (running_) {
        Message msg;
        if (msgrcv(mq_, msg.buf(), BRISBANE_HUB_MQ_MSG_SIZE, 0, 0) == -1) {
            perror("msgrcv");
            continue;
        }
        int header = msg.ReadHeader();
        int pid = msg.ReadPID();
        int ret = BRISBANE_OK;
        switch (header) {
            _debug("header[0x%x]", header);
            case BRISBANE_HUB_MQ_STOP:          ret = ExecuteStop(msg, pid);        break;
            case BRISBANE_HUB_MQ_STATUS:        ret = ExecuteStatus(msg, pid);      break;
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

bool Hub::Running() {
    return access(BRISBANE_HUB_MQ_PATH, F_OK) != -1;
}

int Hub::ExecuteStop(Message& msg, int pid) {
    running_ = false;
    Message fmsg(BRISBANE_HUB_FIFO_STOP);
    SendFIFO(fmsg, pid);
    return BRISBANE_OK;
}

int Hub::ExecuteStatus(Message& msg, int pid) {
    Message fmsg(BRISBANE_HUB_FIFO_STATUS);
    fmsg.WriteInt(ndevs_);
    for (int i = 0; i < ndevs_; i++) {
        fmsg.WriteULong(ntasks_[i]);
    }
    SendFIFO(fmsg, pid);
    return BRISBANE_OK;
}

int Hub::ExecuteRegister(Message& msg, int pid) {
    int ndevs = msg.ReadInt();
    if (ndevs != -1) {
        if (ndevs_ != 0 && ndevs_ != ndevs) _error("ndevs_[%d] ndev[%d]", ndevs_, ndevs);
        ndevs_ = ndevs;
    }
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
        _error("iret[%d][%s]", iret, path);
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

int usage(char** argv) {
    printf("Usage: %s [start | stop | status]\n", argv[0]);
    return -1;
}

int start(char** argv) {
    HubClient* client = new HubClient(NULL);
    client->Init();
    if (client->available()) {
        printf("%s is already running.\n", app);
        delete client;
        return 0;
    }
    delete client;

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 1;
    }
    if (pid > 0) exit(0);

    setsid();
    Hub* hub = new Hub();
    printf("%s is running.\n", app);
    hub->Run();
    delete hub;
    return 0;
}

int stop(char** argv) {
    HubClient* client = new HubClient(NULL);
    client->Init();
    if (client->available()) {
        printf("%s is stopping...", app);
        client->StopHub();
    } else printf("%s is not running...", app);
    delete client;
    char cmd[64];
    memset(cmd, 0, 64);
    sprintf(cmd, "rm -f %s %s*", BRISBANE_HUB_MQ_PATH, BRISBANE_HUB_FIFO_PATH);
    if (system(cmd) == -1) perror(cmd);
    printf("done.\n");
    return 0;
}

int status(char** argv) {
    HubClient* client = new HubClient(NULL);
    client->Init();
    if (!client->available()) {
        printf("%s is not running.\n", app);
    } else {
        printf("%s is running.\n", app);
        client->Status();
    }
    delete client;
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 1) return usage(argv);
    if (strcmp(argv[1], "start") == 0)  return start(argv);
    if (strcmp(argv[1], "stop") == 0)   return stop(argv);
    if (strcmp(argv[1], "status") == 0) return status(argv);
    return usage(argv);
}
