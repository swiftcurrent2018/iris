#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
  brisbane_init(&argc, &argv, true);

  size_t SIZE;
  float *X, *Y, *Z;
  float A = 4;
  int ERROR = 0;

  int nteams = 16;
  int chunk_size = SIZE / nteams;

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  printf("SIZE[%lu]\n", SIZE);

  X = (float*) valloc(SIZE * sizeof(float));
  Y = (float*) valloc(SIZE * sizeof(float));
  Z = (float*) valloc(SIZE * sizeof(float));

  for (int i = 0; i < SIZE; i++) {
    X[i] = 2 * i;
    Y[i] = i;
  }

  brisbane_mem_map(Z, SIZE * sizeof(float));
  brisbane_mem_map(X, SIZE * sizeof(float));
  brisbane_mem_map(Y, SIZE * sizeof(float));

  brisbane_kernel kernel_saxpy;
  brisbane_kernel_create("saxpy", &kernel_saxpy);
  brisbane_kernel_setmap(kernel_saxpy, 0, Z, brisbane_w);
  brisbane_kernel_setarg(kernel_saxpy, 1, sizeof(A), &A);
  brisbane_kernel_setmap(kernel_saxpy, 2, X, brisbane_r);
  brisbane_kernel_setmap(kernel_saxpy, 3, Y, brisbane_r);

  brisbane_task task0;
  brisbane_task_create(&task0);
  brisbane_task_mapto(task0, X, SIZE * sizeof(float));
  brisbane_task_mapto(task0, Y, SIZE * sizeof(float));
  brisbane_task_kernel(task0, kernel_saxpy, 1, NULL, &SIZE);
  brisbane_task_mapfrom(task0, Z, SIZE * sizeof(float));
  brisbane_task_submit(task0, brisbane_random, NULL, true);

  /*
#pragma omp target map(from:Z) map(to:X, Y)
#pragma omp teams num_teams(nteams)
#pragma omp distribute parallel for dist_schedule(static, chunk_size)
  for (int i = 0; i < SIZE; i++) {
    Z[i] = A * X[i] + Y[i];
  }
*/

  for (int i = 0; i < SIZE; i++) {
    //printf("[%8d] %8.1f = %4.0f * %8.1f + %8.1f\n", i, Z[i], A, X[i], Y[i]);
    if (Z[i] != A * X[i] + Y[i]) ERROR++;
  }

  printf("ERROR[%d]\n", ERROR);

  brisbane_mem_unmap(X);
  brisbane_mem_unmap(Y);
  brisbane_mem_unmap(Z);

  free(X);
  free(Y);
  free(Z);

  brisbane_finalize();

  return 0;
}
