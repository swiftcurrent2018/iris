#include <brisbane/brisbane.h>
#include "Utils.h"
#include "Config.h"
#include "Debug.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

namespace brisbane {
namespace rt {

void Utils::Logo(bool color) {
  if (color) {
    srand(time(NULL));
    char str[12];
    sprintf(str, "\033[22;3%dm", rand() % 9 + 1);
    printf("%s", str);
  }
  printf("██████╗ ██████╗ ██╗███████╗██████╗  █████╗ ███╗   ██╗███████╗\n");
  printf("██╔══██╗██╔══██╗██║██╔════╝██╔══██╗██╔══██╗████╗  ██║██╔════╝\n");
  printf("██████╔╝██████╔╝██║███████╗██████╔╝███████║██╔██╗ ██║█████╗  \n");
  printf("██╔══██╗██╔══██╗██║╚════██║██╔══██╗██╔══██║██║╚██╗██║██╔══╝  \n");
  printf("██████╔╝██║  ██║██║███████║██████╔╝██║  ██║██║ ╚████║███████╗\n");
  printf("╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝\n");
  if (color) {
    printf(RESET);
    fprintf(stderr, RESET);
  }
}

void Utils::ReadFile(char* path, char** string, size_t* len) {
  int fd = open((const char*) path, O_RDONLY);
  if (fd == -1) {
    _error("path[%s] %s", path, strerror(errno));
    *len = 0UL;
    return;
  }
  off_t s = lseek(fd, 0, SEEK_END);
  *string = (char*) malloc(s);
  *len = s;
  lseek(fd, 0, SEEK_SET);
  ssize_t r = read(fd, *string, s);
  if (r != s) _error("read[%lu] vs [%lu]", r, s);
  close(fd);
}

} /* namespace rt */
} /* namespace brisbane */
