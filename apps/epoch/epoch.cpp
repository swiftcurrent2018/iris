#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
  size_t SIZE;
  int EPOCH;
  int *A, *B, *C;
  int ERROR = 0;

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  EPOCH = argc > 2 ? atoi(argv[2]) : 4;
  printf("SIZE[%d] EPOCH[%d]\n", SIZE, EPOCH);

  A = (int*) malloc(SIZE * sizeof(int));
  B = (int*) malloc(SIZE * sizeof(int));
  C = (int*) malloc(SIZE * sizeof(int));

  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
    B[i] = i * 1000;
    C[i] = 0;
  }

#pragma omp target data map(to:A[:0], B[:0]) map(from:C[:0])
  for (int e = 0; e < EPOCH; e++) {
#pragma omp target teams distribute parallel for device(brisbane_history)
    for (int i = 0; i < SIZE; i++) {
      C[i] += A[i] + B[i];
    }
  }

  for (int i = 0; i < SIZE; i++) {
    printf("[%8d] %8d = (%8d + %8d) * %d\n", i, C[i], A[i], B[i], EPOCH);
    if (C[i] != (A[i] + B[i]) * EPOCH) ERROR++;
  }
  printf("ERROR[%d]\n", ERROR);

  free(A);
  free(B);
  free(C);

  return 0;
}
