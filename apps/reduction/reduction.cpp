#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
  size_t SIZE;
  int *A;
  size_t sumA, maxA;
  int ERROR = 0;

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  printf("SIZE[%d]\n", SIZE);

  A = (int*) calloc(SIZE, sizeof(int));
  sumA = 0UL;
  maxA = 0UL;

#pragma omp target teams distribute parallel for map(to:A[0:SIZE]) device(gpu)
  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
  }

#pragma omp target teams distribute parallel for reduction(sum:sumA)
  for (int i = 0; i < SIZE; i++) {
    sumA += A[i];
  }

#pragma omp target teams distribute parallel for reduction(max:maxA)
  for (int i = 0; i < SIZE; i++) {
    if (A[i] > maxA) maxA = A[i];
  }

  printf("sumA[%lu] maxA[%lu]\n", sumA, maxA);

  if (sumA != (SIZE - 1) * (SIZE / 2)) ERROR++;
  if (maxA != SIZE - 1) ERROR++;

  printf("ERROR[%d]\n", ERROR);

  free(A);

  return 0;
}
