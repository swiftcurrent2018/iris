#ifndef BRISBANE_RT_SRC_UTILS_H
#define BRISBANE_RT_SRC_UTILS_H

#include <stdio.h>

namespace brisbane {
namespace rt {

class Utils {
public:

static void logo() {
    printf("██████╗ ██████╗ ██╗███████╗██████╗  █████╗ ███╗   ██╗███████╗\n");
    printf("██╔══██╗██╔══██╗██║██╔════╝██╔══██╗██╔══██╗████╗  ██║██╔════╝\n");
    printf("██████╔╝██████╔╝██║███████╗██████╔╝███████║██╔██╗ ██║█████╗  \n");
    printf("██╔══██╗██╔══██╗██║╚════██║██╔══██╗██╔══██║██║╚██╗██║██╔══╝  \n");
    printf("██████╔╝██║  ██║██║███████║██████╔╝██║  ██║██║ ╚████║███████╗\n");
    printf("╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝\n");
}

};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_UTILS_H */
