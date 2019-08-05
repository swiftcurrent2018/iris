#ifndef BRINSBANE_INCLUDE_BRISBANE_BRISBANE_INTERNAL_H
#define BRINSBANE_INCLUDE_BRISBANE_BRISBANE_INTERNAL_H

typedef struct {
  size_t typesz;
  size_t off_r;
  size_t off_w;
  size_t len_r;
  size_t len_w;
  bool   poly_r;
  bool   poly_w;
} brisbane_poly_mem;

#endif /* BRINSBANE_INCLUDE_BRISBANE_BRISBANE_INTERNAL_H */
