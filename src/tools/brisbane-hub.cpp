#include "Hub.h"
#include "HubClient.h"
#include "Debug.h"
#include "Message.h"
#include <stdlib.h>

using namespace brisbane::rt;

namespace brisbane {
namespace rt {

char brisbane_log_prefix_[32] = "BRISBANE-HUB";
const static char* app = "BRISBANE-HUB";

} /* namespace rt */
} /* namespace brisbane */

int usage(char** argv) {
  printf("Usage: %s [ start | stop | restart | status ]\n", argv[0]);
  return 1;
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
  printf("Done.\n");
  return 0;
}

int restart(char** argv) {
  stop(argv);
  return start(argv);
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
  if (strcmp(argv[1], "start") == 0)    return start(argv);
  if (strcmp(argv[1], "stop") == 0)     return stop(argv);
  if (strcmp(argv[1], "restart") == 0)  return restart(argv);
  if (strcmp(argv[1], "status") == 0)   return status(argv);
  return usage(argv);
}

