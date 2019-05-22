#include <brisbane/brisbane.h>
#include "Debug.h"
#include "Platform.h"

using namespace brisbane::rt;

int brisbane_init(int* argc, char*** argv) {
    return Platform::GetPlatform()->Init(argc, argv);
}

int brisbane_finalize() {
    return Platform::Finalize();
}

int brisbane_region_begin(int device_type) {
    return Platform::GetPlatform()->RegionBegin(device_type);
}

int brisbane_mem_create(size_t size, brisbane_mem* mem) {
    return Platform::GetPlatform()->MemCreate(size, mem);
}

int brisbane_mem_h2d(brisbane_mem mem, size_t off, size_t size, void* host) {
    return Platform::GetPlatform()->MemH2D(mem, off, size, host);
}

