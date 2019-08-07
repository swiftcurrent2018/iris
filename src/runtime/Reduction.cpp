#include <brisbane/brisbane.h>
#include "Reduction.h"
#include "Debug.h"
#include "Mem.h"

namespace brisbane {
namespace rt {

Reduction::Reduction() {
  pthread_mutex_init(&mutex_, NULL);
}

Reduction::~Reduction() {
  pthread_mutex_destroy(&mutex_);
}


void Reduction::Reduce(Mem* mem, void* host, size_t size) {
  int mode = mem->mode();
  switch (mode) {
    case brisbane_sum: return Sum(mem, host, size);
  }
  _error("not support mode[0x%x]", mode);
}

void Reduction::Sum(Mem* mem, void* host, size_t size) {
  int type = mem->type();
  if (type == brisbane_long)      return SumLong(mem, (long*) host, size);
  if (type == brisbane_double)    return SumDouble(mem, (double*) host, size);
  _error("not support type[0x%x]", type);
}

void Reduction::SumLong(Mem* mem, long* host, size_t size) {
  long* src = (long*) mem->host_inter();
  long sum = 0;
  for (int i = 0; i < mem->expansion(); i++) sum += src[i]; 
  if (size != sizeof(sum)) _error("size[%lu] sizeof(sum[%lu])", size, sizeof(sum));
#if 1
  __sync_fetch_and_add(host, sum);
#else
  pthread_mutex_lock(&mutex_);
  *host += sum;
  pthread_mutex_unlock(&mutex_);
#endif
}

void Reduction::SumDouble(Mem* mem, double* host, size_t size) {
  double* src = (double*) mem->host_inter();
  double sum = 0.0;
  _debug("mem->expansion[%d]", mem->expansion());
  for (int i = 0; i < mem->expansion(); i++) sum += src[i]; 
  if (size != sizeof(sum)) _error("size[%lu] sizeof(sum[%lu])", size, sizeof(sum));
#if 0
  __sync_fetch_and_add(host, sum);
#else
  pthread_mutex_lock(&mutex_);
  *host += sum;
  pthread_mutex_unlock(&mutex_);
#endif
}

Reduction* Reduction::singleton_ = NULL;

Reduction* Reduction::GetInstance() {
  if (singleton_ == NULL) singleton_ = new Reduction();
  return singleton_;
}

} /* namespace rt */
} /* namespace brisbane */
