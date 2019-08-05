#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
  brisbane_init(&argc, &argv);

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

  brisbane_mem mem_X;
  brisbane_mem mem_Y;
  brisbane_mem mem_Z;
  brisbane_mem_create(SIZE * sizeof(float), &mem_X);
  brisbane_mem_create(SIZE * sizeof(float), &mem_Y);
  brisbane_mem_create(SIZE * sizeof(float), &mem_Z);

  size_t kernel_saxpy_off[1] = { 0 };
  size_t kernel_saxpy_idx[1] = { SIZE };
  brisbane_kernel kernel_saxpy;
  brisbane_kernel_create("saxpy", &kernel_saxpy);
  brisbane_kernel_setmem(kernel_saxpy, 0, mem_Z, brisbane_wr);
  brisbane_kernel_setarg(kernel_saxpy, 1, sizeof(A), &A);
  brisbane_kernel_setmem(kernel_saxpy, 2, mem_X, brisbane_rd);
  brisbane_kernel_setmem(kernel_saxpy, 3, mem_Y, brisbane_rd);

/*
  brisbane_kernel_opt saxpy_opt;
  brisbane_kernel_opt_create(&saxpy_opt);
  brisbane_kernel_opt_num_teams(nteams);
  brisbane_kernel_opt_dist_scheduler(0, chunk_size);
*/

  brisbane_task task0;
  brisbane_task_create(&task0);
  brisbane_task_h2d_full(task0, mem_X, X);
  brisbane_task_h2d_full(task0, mem_Y, Y);
  brisbane_task_kernel(task0, kernel_saxpy, 1, kernel_saxpy_off, kernel_saxpy_idx);
  //brisbane_task_kernel(task0, kernel_saxpy, 1, kernel_saxpy_off, kernel_saxpy_idx, saxpy_opt);
  brisbane_task_d2h_full(task0, mem_Z, Z);
  brisbane_task_submit(task0, brisbane_all | brisbane_cpu | brisbane_gpu, NULL, true);
/*
#pragma omp target map(from:Z) map(to:X, Y)
#pragma omp teams num_teams(nteams)
#pragma omp distribute parallel for dist_schedule(static, chunk_size)
  for (int i = 0; i < SIZE; i++) {
    Z[i] = A * X[i] + Y[i];
  }
*/

  for (int i = 0; i < SIZE; i++) {
    printf("[%8d] %8.1f = %4.0f * %8.1f + %8.1f\n", i, Z[i], A, X[i], Y[i]);
    if (Z[i] != A * X[i] + Y[i]) ERROR++;
  }

  printf("ERROR[%d]\n", ERROR);

  free(X);
  free(Y);
  free(Z);

  brisbane_finalize();

  return 0;
}
