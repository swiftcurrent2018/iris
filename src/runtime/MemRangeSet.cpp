#include "MemRangeSet.h"
#include "Debug.h"

namespace brisbane {
namespace rt {

MemRangeSet::MemRangeSet() {
}

MemRangeSet::~MemRangeSet() {
}

void MemRangeSet::Add(size_t off, size_t size) {
  MemRange range = { off, size };
  ranges_.insert(range);
}

bool MemRangeSet::Contain(size_t off, size_t size) {
  for (std::set<MemRange>::iterator I = ranges_.begin(), E = ranges_.end(); I != E; ++I) {
    MemRange r = *I;
    if (r.off > off) return false;
    if (r.off <= off && r.off + r.size >= off + size) return true;
  }
  return false;
}

} /* namespace rt */
} /* namespace brisbane */
