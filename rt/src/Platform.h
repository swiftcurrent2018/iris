#ifndef BRISBANE_RT_SRC_PLATFORM_H
#define BRISBANE_RT_SRC_PLATFORM_H

#include <brisbane/brisbane.h>
#include <CL/cl.h>
#include <stddef.h>
#include <set>
#include "Debug.h"

namespace brisbane {
namespace rt {

class Device;
class Mem;

class Platform {
private:
    Platform();
    ~Platform();

public:
    int Init(int* argc, char*** argv);
    int GetCLPlatforms();

    int RegionBegin(int device_type);

    int MemCreate(size_t size, brisbane_mem* brs_mem);
    int MemH2D(brisbane_mem mem, size_t off, size_t size, void* host);

public:
    static Platform* GetPlatform();
    static int Finalize();

private:
    bool init_;

    Device* devices_[16];
    int num_devices_;

    cl_platform_id cl_platforms_[16];
    cl_device_id cl_devices_[16];
    cl_int clerr;

    std::set<Mem*> mems_;
private:
    static Platform* singleton_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_PLATFORM_H */
