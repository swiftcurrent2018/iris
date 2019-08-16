#ifndef BRISBANE_RT_SRC_MEM_RANGE_SET_H
#define BRISBANE_RT_SRC_MEM_RANGE_SET_H

#include <stddef.h>
#include <set>

namespace brisbane {
namespace rt {

typedef struct MemRange_ {
  size_t off;
  size_t size;
  bool operator <(const MemRange_& m) const { return off < m.off; }
} MemRange;

class MemRangeSet {
public:
  MemRangeSet();
  ~MemRangeSet();

  bool Empty() { return ranges_.empty(); }
  size_t Size() { return ranges_.size(); }
  void Clear() { return ranges_.clear(); }

  void Add(size_t off, size_t size);
  bool Contain(size_t off, size_t size);

private:
  std::set<MemRange> ranges_;

};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_MEM_RANGE_SET_H */

