#ifndef BRISBANE_RT_SRC_MEM_H
#define BRISBANE_RT_SRC_MEM_H

#include "Object.h"
#include "Platform.h"

namespace brisbane {
namespace rt {

class Mem: public Object<struct _brisbane_mem, Mem> {
public:
    Mem(size_t size);
    virtual ~Mem();

    void AddOwner(Device* dev);
    void SetOwner(Device* dev);
    bool IsOwner(Device* dev);

    cl_mem clmem(int i, cl_context clctx);

    size_t size() { return size_; }
    Device* owner() { return owners_[0]; }
    void* host_inter();

private:
    size_t size_;
    cl_mem clmems_[16];
    cl_int clerr_;
    Device* owners_[16];
    int owners_num_;
    void* host_inter_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_MEM_H */
