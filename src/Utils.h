#ifndef BRISBANE_RT_SRC_UTILS_H
#define BRISBANE_RT_SRC_UTILS_H

#include <stdlib.h>

namespace brisbane {
namespace rt {

class Utils {
public:
static void Logo(bool color);
static void ReadFile(char* path, char** string, size_t* len);
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_UTILS_H */
