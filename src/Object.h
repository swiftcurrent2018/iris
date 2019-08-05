#ifndef BRISBANE_RT_OBJECT_H
#define BRISBANE_RT_OBJECT_H

#include <brisbane/brisbane.h>
#include "Structs.h"
#include <stddef.h>

extern unsigned long brisbane_create_new_uid();

namespace brisbane {
namespace rt {

template <typename struct_type, class class_type>
class Object {
public:
  Object() {
    uid_ = brisbane_create_new_uid();
    struct_obj_.class_obj = (class_type*) this;
  }
  virtual ~Object() {}

  unsigned long uid() { return uid_; }
  struct_type* struct_obj() { return &struct_obj_; }

private:
  unsigned long uid_;
  struct_type struct_obj_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_OBJECT_H */
