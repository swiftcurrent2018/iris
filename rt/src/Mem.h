#ifndef BRISBANE_RT_SRC_MEM_H
#define BRISBANE_RT_SRC_MEM_H

#include "Object.h"
#include "Structs.h"

namespace brisbane {
namespace rt {

class Mem: public Object<struct _brisbane_mem, Mem> {
public:
    Mem(size_t size);
    virtual ~Mem();

private:
    size_t size_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_MEM_H */
