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

    cl_mem clmem(int i, cl_context clctx);

private:
    size_t size_;
    cl_mem clmems_[16];
    cl_int clerr_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_MEM_H */
