#ifndef BRINSBANE_INCLUDE_BRISBANE_BRINSBANE_H
#define BRINSBANE_INCLUDE_BRISBANE_BRINSBANE_H

#define BRISBANE_OK             0
#define BRISBANE_ERR            -1

#ifdef __cplusplus
extern "C" {
#endif

extern int brisbane_init(int* argc, char*** argv);
extern int brisbane_finalize();

#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif /* BRINSBANE_INCLUDE_BRISBANE_BRINSBANE_H */
