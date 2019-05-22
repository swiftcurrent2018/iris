#include "Platform.h"
#include "Utils.h"
#include "Device.h"
#include "Mem.h"
#include <unistd.h>

namespace brisbane {
namespace rt {

char debug_prefix_[256];

Platform::Platform() {
    init_ = false;
    num_devices_ = 0;
}

Platform::~Platform() {
    _check();
    if (!init_) return;
}

int Platform::Init(int* argc, char*** argv) {
    if (init_) return BRISBANE_ERR;
    gethostname(debug_prefix_, 256);
    Utils::logo();
    _check();
    GetCLPlatforms();
    return BRISBANE_OK;
}

int Platform::GetCLPlatforms() {
    cl_uint num_platforms;
    cl_uint num_devices;

    clerr = clGetPlatformIDs(0, NULL, &num_platforms);
    _trace("num_platforms[%u]", num_platforms);
    clerr = clGetPlatformIDs(num_platforms, cl_platforms_, NULL);
    char platform_vendor[64];
    char platform_name[64];
    for (cl_uint i = 0; i < num_platforms; i++) {
        clerr = clGetPlatformInfo(cl_platforms_[i], CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, NULL);
        clerr = clGetPlatformInfo(cl_platforms_[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        clerr = clGetDeviceIDs(cl_platforms_[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        clerr = clGetDeviceIDs(cl_platforms_[i], CL_DEVICE_TYPE_ALL, num_devices, cl_devices_ + num_devices_, NULL);
        _trace("OpenCL Platform[%d] %s / %s : num_devices[%u]", i, platform_vendor, platform_name, num_devices);
        for (cl_uint j = 0; j < num_devices; j++) {
            devices_[num_devices_] = new Device(cl_devices_[num_devices_++]);
        }
    }
}

int Platform::RegionBegin(int device_type) {
    _check();
    return BRISBANE_OK;
}

int Platform::MemCreate(size_t size, brisbane_mem* brs_mem) {
    _check();
    Mem* mem = new Mem(size);
    *brs_mem = mem->struct_obj();
    mems_.insert(mem);
    return BRISBANE_OK;
}

int Platform::MemH2D(brisbane_mem mem, size_t off, size_t size, void* host) {
    _check();
    return BRISBANE_OK;
}

Platform* Platform::singleton_ = NULL;

Platform* Platform::GetPlatform() {
    if (singleton_ == NULL) singleton_ = new Platform();
    return singleton_;
}

int Platform::Finalize() {
    _check();
    if (singleton_ == NULL) return BRISBANE_ERR;
    if (singleton_) delete singleton_;
    singleton_ = NULL;
    return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */
