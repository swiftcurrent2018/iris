#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
  size_t SIZE;
  float *X, *Y, *Z;
  float A = 4;
  int ERROR = 0;

  int nteams = 16;
  int chunk_size = SIZE / nteams;

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  printf("SIZE[%d]\n", SIZE);

  X = (float*) malloc(SIZE * sizeof(float));
  Y = (float*) malloc(SIZE * sizeof(float));
  Z = (float*) malloc(SIZE * sizeof(float));

  for (int i = 0; i < SIZE; i++) {
    X[i] = 2 * i;
    Y[i] = i;
  }

#pragma omp target map(from:Z) map(to:X, Y)
#pragma omp teams num_teams(nteams)
#pragma omp distribute parallel for dist_schedule(static, chunk_size)
  for (int i = 0; i < SIZE; i++) {
    Z[i] = A * X[i] + Y[i];
  }

  for (int i = 0; i < SIZE; i++) {
    printf("[%8d] %8.1f = %4.0f * %8.1f + %8.1f\n", i, Z[i], A, X[i], Y[i]);
    if (Z[i] != A * X[i] + Y[i]) ERROR++;
  }

  printf("ERROR[%d]\n", ERROR);

  free(X);
  free(Y);
  free(Z);

  return 0;
}
