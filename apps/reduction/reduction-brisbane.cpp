#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
  brisbane_init(&argc, &argv, true);

  size_t SIZE;
  int *A;
  size_t sumA, maxA;
  int ERROR = 0;

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  printf("SIZE[%d]\n", SIZE);

  A = (int*) calloc(SIZE, sizeof(int));
  sumA = 0UL;
  maxA = 0UL;

  brisbane_mem mem_A;
  brisbane_mem_create(SIZE * sizeof(int), &mem_A);

  brisbane_kernel kernel_init;
  brisbane_kernel_create("init", &kernel_init);
  brisbane_kernel_setmem(kernel_init, 0, mem_A, brisbane_wr);

  brisbane_task task0;
  brisbane_task_create(&task0);
  size_t kernel_init_off[1] = { 0 };
  size_t kernel_init_idx[1] = { SIZE };
  brisbane_task_kernel(task0, kernel_init, 1, kernel_init_off, kernel_init_idx);
  brisbane_task_submit(task0, brisbane_cpu, NULL, true);
  /*
#pragma omp target teams distribute parallel for map(to:A[0:SIZE]) device(gpu)
  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
  }
  */

  brisbane_mem mem_sumA;
  brisbane_mem_create(sizeof(size_t), &mem_sumA);
  brisbane_mem_reduce(mem_sumA, brisbane_sum, brisbane_long);

  brisbane_kernel kernel_sum;
  brisbane_kernel_create("reduce_sum", &kernel_sum);
  size_t kernel_sum_off[1] = { 0 };
  size_t kernel_sum_idx[1] = { SIZE };
  brisbane_kernel_setmem(kernel_sum, 0, mem_A, brisbane_rd);
  brisbane_kernel_setmem(kernel_sum, 1, mem_sumA, brisbane_wr);

  brisbane_task task1;
  brisbane_task_create(&task1);
  brisbane_task_kernel(task1, kernel_sum, 1, kernel_sum_off, kernel_sum_idx);
  brisbane_task_d2h(task1, mem_sumA, 0, sizeof(size_t), &sumA);
  brisbane_task_submit(task1, brisbane_data, NULL, true);
  /*
#pragma omp target teams distribute parallel for reduction(+:sumA)
for (int i = 0; i < SIZE; i++) {
sumA += A[i];
}
*/

  brisbane_mem mem_maxA;
  brisbane_mem_create(sizeof(size_t), &mem_maxA);

  brisbane_kernel kernel_max;
  brisbane_kernel_create("reduce_max", &kernel_max);
  brisbane_kernel_setmem(kernel_max, 0, mem_A, brisbane_rd);
  brisbane_kernel_setmem(kernel_max, 1, mem_maxA, brisbane_wr);

  brisbane_task task2;
  brisbane_task_create(&task2);
  size_t kernel_max_off[1] = { 0 };
  size_t kernel_max_idx[1] = { SIZE };
  brisbane_task_kernel(task2, kernel_max, 1, kernel_max_off, kernel_max_idx);
  brisbane_task_d2h(task2, mem_maxA, 0, sizeof(size_t), &maxA);
  brisbane_task_submit(task2, brisbane_data, NULL, true);
/*
#pragma omp target teams distribute parallel for reduction(max:maxA)
  for (int i = 0; i < SIZE; i++) {
    if (A[i] > maxA) maxA = A[i];
  }
*/

  size_t sum = 0;
  for (size_t i = 0; i < SIZE; i++) sum += i;
  if (sumA != sum) ERROR++;
  if (maxA != SIZE - 1) ERROR++;

  printf("ERROR[%d] sum[%lu] sumA[%lu] maxA[%lu]\n", ERROR, sum, sumA, maxA);

  free(A);

  brisbane_task_release(task0);
  brisbane_kernel_release(kernel_init);
  brisbane_mem_release(mem_A);

  brisbane_finalize();

  return 0;
}
