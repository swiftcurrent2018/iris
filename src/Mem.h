#ifndef BRISBANE_RT_SRC_MEM_H
#define BRISBANE_RT_SRC_MEM_H

#include "Object.h"
#include "Platform.h"

namespace brisbane {
namespace rt {

typedef struct _MemRange {
    size_t off;
    size_t size;
    struct _MemRange* next;
} MemRange;

class Mem: public Object<struct _brisbane_mem, Mem> {
public:
    Mem(size_t size, Platform* platform);
    virtual ~Mem();

    void SetOwner(Device* dev);
    bool IsOwner(Device* dev);
    void AddOwner(size_t off, size_t size, Device* dev);
    bool IsOwner(size_t off, size_t size, Device* dev);
    Device* owner();
    void Reduce(int mode, int type);
    void Expand(int expansion);

    cl_mem clmem(int i, cl_context clctx);

    size_t size() { return size_; }
    int mode() { return mode_; }
    int type() { return type_; }
    int type_size() { return type_size_; }
    int expansion() { return expansion_; }
    void* host_inter();

private:
    void MemRangeInit(MemRange* range);

private:
    size_t size_;
    int mode_;
    Platform* platform_;
    cl_mem clmems_[BRISBANE_MAX_NDEVS];
    MemRange ranges_[BRISBANE_MAX_NDEVS];
    int nowners_;
    cl_int clerr_;
    void* host_inter_;
    int ndevs_;
    int type_;
    int type_size_;
    int expansion_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_MEM_H */
