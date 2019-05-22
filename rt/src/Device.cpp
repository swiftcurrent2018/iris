#include "Device.h"

namespace brisbane {
namespace rt {

Device::Device(cl_device_id cl_device) {
    cl_device_ = cl_device;
    clerr = clGetDeviceInfo(cl_device_, CL_DEVICE_VENDOR, sizeof(vendor_), vendor_, NULL);
    clerr = clGetDeviceInfo(cl_device_, CL_DEVICE_NAME, sizeof(name_), name_, NULL);
    clerr = clGetDeviceInfo(cl_device_, CL_DEVICE_TYPE, sizeof(type_), &type_, NULL);
    clerr = clGetDeviceInfo(cl_device_, CL_DEVICE_VERSION, sizeof(version_), version_, NULL);
    _info("Device[%s:%s] type[0x%x] version[%s]", vendor_, name_, type_, version_);
}

Device::~Device() {
    _check();
}

} /* namespace rt */
} /* namespace brisbane */
