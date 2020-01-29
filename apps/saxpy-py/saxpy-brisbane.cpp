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

  int nteams = 8;
  int chunk_size = SIZE / nteams;

  SIZE = argc > 1 ? atol(argv[1]) : 8;
  printf("SIZE[%lu]\n", SIZE);

  X = (float*) malloc(SIZE * sizeof(float));
  Y = (float*) malloc(SIZE * sizeof(float));
  Z = (float*) malloc(SIZE * sizeof(float));

  for (int i = 0; i < SIZE; i++) {
    X[i] = 2 * i;
    Y[i] = i;
  }

  brisbane_mem mem_X;
  brisbane_mem mem_Y;
  brisbane_mem mem_Z;
  brisbane_mem_create(SIZE * sizeof(float), &mem_X);
  brisbane_mem_create(SIZE * sizeof(float), &mem_Y);
  brisbane_mem_create(SIZE * sizeof(float), &mem_Z);

  size_t kernel_saxpy0_off[1] = { 0 };
  size_t kernel_saxpy0_idx[1] = { SIZE };
  brisbane_kernel kernel_saxpy0;
  brisbane_kernel_create("saxpy0", &kernel_saxpy0);
  brisbane_kernel_setmem(kernel_saxpy0, 0, mem_Z, brisbane_w);
  brisbane_kernel_setarg(kernel_saxpy0, 1, sizeof(A), &A);
  brisbane_kernel_setmem(kernel_saxpy0, 2, mem_X, brisbane_r);

  brisbane_task task0;
  brisbane_task_create(&task0);
  brisbane_task_h2d_full(task0, mem_X, X);
  brisbane_task_kernel(task0, kernel_saxpy0, 1, kernel_saxpy0_off, kernel_saxpy0_idx);
  brisbane_task_submit(task0, brisbane_cpu, NULL, true);

  size_t kernel_saxpy1_off[1] = { 0 };
  size_t kernel_saxpy1_idx[1] = { SIZE };
  brisbane_kernel kernel_saxpy1;
  brisbane_kernel_create("saxpy1", &kernel_saxpy1);
  brisbane_kernel_setmem(kernel_saxpy1, 0, mem_Z, brisbane_rw);
  brisbane_kernel_setmem(kernel_saxpy1, 1, mem_Y, brisbane_r);

  brisbane_task task1;
  brisbane_task_create(&task1);
  brisbane_task_h2d_full(task1, mem_Y, Y);
  brisbane_task_kernel(task1, kernel_saxpy1, 1, kernel_saxpy1_off, kernel_saxpy1_idx);
  brisbane_task_d2h_full(task1, mem_Z, Z);
  brisbane_task_submit(task1, brisbane_cpu, NULL, true);

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

  brisbane_mem_release(mem_X);
  brisbane_mem_release(mem_Y);
  brisbane_mem_release(mem_Z);

  free(X);
  free(Y);
  free(Z);

  brisbane_finalize();

  return 0;
}
