#ifndef BRISBANE_RT_SRC_UTILS_H
#define BRISBANE_RT_SRC_UTILS_H

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

namespace brisbane {
namespace rt {

class Utils {
public:
static void Logo(bool color) {
    if (color) {
        srand(time(NULL));
        char str[12];
        sprintf(str, "\033[22;3%dm", rand() % 9 + 1);
        printf(str);
    }
    printf("██████╗ ██████╗ ██╗███████╗██████╗  █████╗ ███╗   ██╗███████╗\n");
    printf("██╔══██╗██╔══██╗██║██╔════╝██╔══██╗██╔══██╗████╗  ██║██╔════╝\n");
    printf("██████╔╝██████╔╝██║███████╗██████╔╝███████║██╔██╗ ██║█████╗  \n");
    printf("██╔══██╗██╔══██╗██║╚════██║██╔══██╗██╔══██║██║╚██╗██║██╔══╝  \n");
    printf("██████╔╝██║  ██║██║███████║██████╔╝██║  ██║██║ ╚████║███████╗\n");
    printf("╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝\n");
    if (color) printf("\e[m");
}

static void ReadFile(char* path, char** string, size_t* len) {
    int fd = open((const char*) path, O_RDONLY);
    if (fd == -1) {
        _error("path[%s] %s", path, strerror(errno));
        *len = 0UL;
        return;
    }
    off_t s = lseek(fd, 0, SEEK_END);
    *string = (char*) malloc(s + 1);
    *len = s + 1;
    (*string)[s] = 0;
    lseek(fd, 0, SEEK_SET);
    ssize_t r = read(fd, *string, s);
    close(fd);
}

};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_UTILS_H */
