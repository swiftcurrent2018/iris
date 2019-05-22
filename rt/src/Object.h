#ifndef BRISBANE_RT_OBJECT_H
#define BRISBANE_RT_OBJECT_H

#include <stddef.h>

namespace brisbane {
namespace rt {

template <typename struct_type, class class_type>
class Object {
public:
    Object() {
        struct_obj_.class_obj = (class_type*) this;
    }
    virtual ~Object() {}

    struct_type* struct_obj() { return &struct_obj_; }

private:
    struct_type struct_obj_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_OBJECT_H */
