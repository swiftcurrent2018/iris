#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  size_t SIZE;
  int *A, *B;

  brisbane_init(&argc, &argv, true);

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  printf("SIZE[%d]\n", SIZE);

  A = (int*) valloc(SIZE * sizeof(int));
  B = (int*) valloc(SIZE * sizeof(int));

  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
    B[i] = i * 1000;
  }

  brisbane_mem mem_A;
  brisbane_mem mem_B;
  /*
  brisbane_mem_map(mem_A, A, SIZE * sizeof(int));
  brisbane_mem_map(mem_B, B, SIZE * sizeof(int));
  */
  brisbane_mem_create(SIZE * sizeof(int), &mem_A);
  brisbane_mem_create(SIZE * sizeof(int), &mem_B);
#pragma omp target data map(A, B)
  {

  brisbane_kernel kernel0;
  brisbane_kernel_create("loop0", &kernel0);
  brisbane_kernel_setmem(kernel0, 0, mem_A, brisbane_rw);

  brisbane_task task0;
  brisbane_task_create(&task0);
  brisbane_task_h2d_full(task0, mem_A, A);
  size_t kernel_loop0_off[1] = { 0 };
  size_t kernel_loop0_idx[1] = { SIZE };
  brisbane_task_kernel(task0, kernel0, 1, kernel_loop0_off, kernel_loop0_idx);
  brisbane_task_submit(task0, brisbane_gpu, NULL, true);
  brisbane_task_release(task0);
  brisbane_kernel_release(kernel0);
#if 0
#pragma omp parallel for
  for (int i = 0; i < SIZE; i++) {
    A[i] *= 2;
  }
#endif

  brisbane_kernel kernel1;
  brisbane_kernel_create("loop1", &kernel1);
  brisbane_kernel_setmem(kernel1, 0, mem_B, brisbane_rw);
  brisbane_kernel_setmem(kernel1, 1, mem_A, brisbane_r);

  brisbane_task task1;
  brisbane_task_create(&task1);
  size_t kernel_loop1_off[1] = { 0 };
  size_t kernel_loop1_idx[1] = { SIZE };
  brisbane_task_h2d_full(task1, mem_B, B);
  brisbane_task_kernel(task1, kernel1, 1, kernel_loop1_off, kernel_loop1_idx);
  brisbane_task_d2h_full(task1, mem_B, B);
  brisbane_task_submit(task1, brisbane_gpu, NULL, true);
  brisbane_task_release(task1);
  brisbane_kernel_release(kernel1);
#if 0
#pragma omp parallel for
  for (int i = 0; i < SIZE; i++) {
    B[i] += A[i];
  }
#endif

  for (int i = 0; i < SIZE; i++) {
    printf("[%8d] A[%8d] B[%8d]\n", i, A[i], B[i]);
  }
  }

  brisbane_mem_release(mem_A);
  brisbane_mem_release(mem_B);

  free(A);
  free(B);

  brisbane_finalize();

  return 0;
}
