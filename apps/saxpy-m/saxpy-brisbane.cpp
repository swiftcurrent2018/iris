#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
  brisbane_init(&argc, &argv, true);

  size_t SIZE = argc > 1 ? atol(argv[1]) : 16;
  float *X, *Y, *Z;
  float A = 4;
  int ERROR = 0;

  int ndevs;

  brisbane_info_ndevs(&ndevs);

  int nteams = ndevs * 2;
  int chunk_size = SIZE / nteams;

  printf("SIZE[%d] ndevs[%d]\n", SIZE, ndevs);

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
  brisbane_kernel_setmem(kernel_saxpy, 0, mem_Z, brisbane_w);
  brisbane_kernel_setarg(kernel_saxpy, 1, sizeof(A), &A);
  brisbane_kernel_setmem(kernel_saxpy, 2, mem_X, brisbane_r);
  brisbane_kernel_setmem(kernel_saxpy, 3, mem_Y, brisbane_r);

  brisbane_task task0;
  brisbane_task_create(&task0);
  for (int i = 0; i < SIZE; i += chunk_size) {
    brisbane_task subtask;
    brisbane_task_create(&subtask);
    brisbane_task_h2d(subtask, mem_X, i * sizeof(float), chunk_size * sizeof(float), X + i);
    brisbane_task_h2d(subtask, mem_Y, i * sizeof(float), chunk_size * sizeof(float), Y + i);
    size_t kernel_saxpy_off[1] = { i };
    size_t kernel_saxpy_idx[1] = { chunk_size };
    brisbane_task_kernel(subtask, kernel_saxpy, 1, kernel_saxpy_off, kernel_saxpy_idx);
    brisbane_task_d2h(subtask, mem_Z, i * sizeof(float), chunk_size * sizeof(float), Z + i);
    brisbane_task_add_subtask(task0, subtask);
  }
  brisbane_task_submit(task0, brisbane_any, NULL, true);
/*
  #pragma omp target
  #pragma omp teams num_teams(nteams)
  #pragma omp distribute
  for (int i = 0; i < SIZE; i += chunk_size) {
  #pragma omp target data map(to: X[i*chunk_size:chunk_size], Y[i*chunk_size:chunk_size]) map(from: Z[i*chunk_size:chunk_size])
  #pragma omp parallel for
    for (int j = i; j < i + chunk_size; j++) {
      Z[j] = A * X[j] + Y[j];
    }
  }
*/

  for (int i = 0; i < SIZE; i++) {
    //printf("[%8d] %8.1f = %4.0f * %8.1f + %8.1f\n", i, Z[i], A, X[i], Y[i]);
    if (Z[i] != A * X[i] + Y[i]) ERROR++;
  }

  printf("ERROR[%d]\n", ERROR);

  free(X);
  free(Y);
  free(Z);

  brisbane_finalize();

  return 0;
}
