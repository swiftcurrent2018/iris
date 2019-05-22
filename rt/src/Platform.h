#ifndef BRISBANE_RT_PLATFORM_H
#define BRISBANE_RT_PLATFORM_H

#include <brisbane/brisbane.h>
#include <stddef.h>

namespace brisbane {
namespace rt {

class Platform {
private:
    Platform();
    ~Platform();

public:
    int Init(int* argc, char*** argv);

public:
    static Platform* GetPlatform();
    static int Finalize();

private:
    bool init_;

private:
    static Platform* singleton_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_PLATFORM_H */
