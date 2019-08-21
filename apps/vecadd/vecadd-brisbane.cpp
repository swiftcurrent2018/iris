#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
  size_t SIZE;
  int *A, *B, *C, *D, *E;
  int ERROR = 0;

  brisbane_init(&argc, &argv, true);

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  printf("SIZE[%d]\n", SIZE);

  A = (int*) valloc(SIZE * sizeof(int));
  B = (int*) valloc(SIZE * sizeof(int));
  C = (int*) valloc(SIZE * sizeof(int));
  D = (int*) valloc(SIZE * sizeof(int));
  E = (int*) valloc(SIZE * sizeof(int));

  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
    B[i] = i * 1000;
  }

  brisbane_mem mem_A;
  brisbane_mem mem_B;
  brisbane_mem mem_C;
  brisbane_mem_create(SIZE * sizeof(int), &mem_A);
  brisbane_mem_create(SIZE * sizeof(int), &mem_B);
  brisbane_mem_create(SIZE * sizeof(int), &mem_C);

  brisbane_kernel kernel_loop0;
  brisbane_kernel_create("loop0", &kernel_loop0);
  brisbane_kernel_setmem(kernel_loop0, 0, mem_C, brisbane_w);
  brisbane_kernel_setmem(kernel_loop0, 1, mem_A, brisbane_r);
  brisbane_kernel_setmem(kernel_loop0, 2, mem_B, brisbane_r);

  brisbane_task task0;
  brisbane_task_create_name("loop0", &task0);
  brisbane_task_h2d_full(task0, mem_A, A);
  brisbane_task_h2d_full(task0, mem_B, B);
  size_t kernel_loop0_off[1] = { 0 };
  size_t kernel_loop0_idx[1] = { SIZE };
  brisbane_task_kernel(task0, kernel_loop0, 1, kernel_loop0_off, kernel_loop0_idx);
  brisbane_task_submit(task0, brisbane_all, NULL, true);
/*
#pragma acc parallel loop copyin(A[0:SIZE], B[0:SIZE]) device(gpu)
#pragma omp target teams distribute parallel for map(to:A[0:SIZE], B[0:SIZE]) device(gpu)
#pragma brisbane kernel h2d(A[0:SIZE], B[0:SIZE]) alloc(C[0:SIZE]) device(gpu)
  for (int i = 0; i < SIZE; i++) {
    C[i] = A[i] + B[i];
  }
*/

  brisbane_mem mem_D;
  brisbane_mem_create(SIZE * sizeof(int), &mem_D);

  brisbane_kernel kernel_loop1;
  brisbane_kernel_create("loop1", &kernel_loop1);
  brisbane_kernel_setmem(kernel_loop1, 0, mem_D, brisbane_w);
  brisbane_kernel_setmem(kernel_loop1, 1, mem_C, brisbane_r);

  brisbane_task task1;
  brisbane_task_create_name("loop1", &task1);
  size_t kernel_loop1_off[1] = { 0 };
  size_t kernel_loop1_idx[1] = { SIZE };
  brisbane_task_kernel(task1, kernel_loop1, 1, kernel_loop1_off, kernel_loop1_idx);
  brisbane_task_submit(task1, brisbane_all, NULL, true);
/*
  #pragma acc parallel loop present(C[0:SIZE]) device(cpu)
  #pragma omp target teams distribute parallel for device(cpu)
  #pragma brisbane kernel present(C[0:SIZE]) device(cpu)
  for (int i = 0; i < SIZE; i++) {
    D[i] = C[i] * 10;
  }
*/

  brisbane_mem mem_E;
  brisbane_mem_create(SIZE * sizeof(int), &mem_E);

  brisbane_kernel kernel_loop2;
  brisbane_kernel_create("loop2", &kernel_loop2);
  brisbane_kernel_setmem(kernel_loop2, 0, mem_E, brisbane_w);
  brisbane_kernel_setmem(kernel_loop2, 1, mem_D, brisbane_r);

  brisbane_task task2;
  brisbane_task_create_name("loop2", &task2);
  size_t kernel_loop2_off[1] = { 0 };
  size_t kernel_loop2_idx[1] = { SIZE };
  brisbane_task_kernel(task2, kernel_loop2, 1, kernel_loop2_off, kernel_loop2_idx);
  brisbane_task_d2h_full(task2, mem_E, E);
  brisbane_task_submit(task2, brisbane_all, NULL, true);
/*
#pragma acc parallel loop present(D[0:SIZE]) device(data)
#pragma omp target teams distribute parallel for map(from:E[0:SIZE]) device(data)
#pragma brisbane kernel d2h(E[0:SIZE]) present(D[0:SIZE]) device(data)
  for (int i = 0; i < SIZE; i++) {
    E[i] = D[i] * 2;
  }
*/

  for (int i = 0; i < SIZE; i++) {
    printf("[%8d] %8d = (%8d + %8d) * %d\n", i, E[i], A[i], B[i], 20);
    if (E[i] != (A[i] + B[i]) * 20) ERROR++;
  }
  printf("ERROR[%d]\n", ERROR);

  brisbane_task_release(task0);
  brisbane_task_release(task1);
  brisbane_task_release(task2);
  brisbane_kernel_release(kernel_loop0);
  brisbane_kernel_release(kernel_loop1);
  brisbane_kernel_release(kernel_loop2);
  brisbane_mem_release(mem_A);
  brisbane_mem_release(mem_B);
  brisbane_mem_release(mem_C);
  brisbane_mem_release(mem_D);
  brisbane_mem_release(mem_E);

  free(A);
  free(B);
  free(C);
  free(D);
  free(E);

  brisbane_finalize();

  return 0;
}
