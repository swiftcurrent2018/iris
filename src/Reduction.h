#ifndef BRISBANE_RT_SRC_REDUCTION_H
#define BRISBANE_RT_SRC_REDUCTION_H

#include <stdlib.h>

namespace brisbane {
namespace rt {

class Mem;

class Reduction {
private:
    Reduction();
    ~Reduction();

public:
    void Reduce(Mem* mem, void* host, size_t size);

private:
    void Sum(Mem* mem, void* host, size_t size);
    void SumLong(Mem* mem, void* host, size_t size);

public:
    static Reduction* GetInstance();

private:
    static Reduction* singleton_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_RT_SRC_REDUCTION_H */
