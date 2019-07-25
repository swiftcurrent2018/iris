#ifndef BRINSBANE_INCLUDE_BRISBANE_BRISBANE_INTERNAL_H
#define BRINSBANE_INCLUDE_BRISBANE_BRISBANE_INTERNAL_H

typedef struct {
    size_t typesz;
    size_t offset;
    size_t length;
    int    rwmode;
    bool   enable;
} brisbane_poly_mem;

#endif /* BRINSBANE_INCLUDE_BRISBANE_BRISBANE_INTERNAL_H */
