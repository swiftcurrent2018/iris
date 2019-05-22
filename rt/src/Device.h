#ifndef BRISBANE_RT_SRC_DEVICE_H
#define BRISBANE_RT_SRC_DEVICE_H

#include "Platform.h"

namespace brisbane {
namespace rt {

class Device {
public:
    Device(cl_device_id cl_device);
    ~Device();

private:
    cl_device_id cl_device_;
    cl_device_type type_;
    char vendor_[64];
    char name_[64];
    char version_[64];
    cl_int clerr;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_DEVICE_H */
