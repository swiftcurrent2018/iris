#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
  size_t SIZE;
  int EPOCH;
  int *A, *B, *C;
  int ERROR = 0;

  brisbane_init(&argc, &argv, true);

  int nplatforms;
  int ndevs;

  brisbane_platform_count(&nplatforms);
  brisbane_device_count(&ndevs);

  printf("[%s:%d] nplatforms[%d] ndevs[%d]\n", __FILE__, __LINE__, nplatforms, ndevs);

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  EPOCH = argc > 2 ? atoi(argv[2]) : 4;
  printf("SIZE[%d] EPOCH[%d]\n", SIZE, EPOCH);

  A = (int*) valloc(SIZE * sizeof(int));
  B = (int*) valloc(SIZE * sizeof(int));
  C = (int*) valloc(SIZE * sizeof(int));

  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
    B[i] = i * 1000;
    C[i] = 0;
  }

  int current_dev;
  brisbane_device_get_default(&current_dev);
  printf("[%s:%d] current_dev[0x%x]\n", __FILE__, __LINE__,  current_dev);

  brisbane_mem mem_A;
  brisbane_mem mem_B;
  brisbane_mem mem_C;
  brisbane_mem_create(SIZE * sizeof(int), &mem_A);
  brisbane_mem_create(SIZE * sizeof(int), &mem_B);
  brisbane_mem_create(SIZE * sizeof(int), &mem_C);

  brisbane_task task0;
  brisbane_task_create(&task0);
  brisbane_task_h2d_full(task0, mem_A, A);
  brisbane_task_h2d_full(task0, mem_B, B);
  brisbane_task_h2d_full(task0, mem_C, C);
  brisbane_task_submit(task0, brisbane_cpu, NULL, true);

  brisbane_kernel kernel_loop0;
  brisbane_kernel_create("loop0", &kernel_loop0);
  brisbane_kernel_setmem(kernel_loop0, 0, mem_C, brisbane_rw);
  brisbane_kernel_setmem(kernel_loop0, 1, mem_A, brisbane_r);
  brisbane_kernel_setmem(kernel_loop0, 2, mem_B, brisbane_r);

#pragma brisbane data h2d(A[0:SIZE], B[0:SIZE], C[0:SIZE]) d2h(C[0:SIZE])
  for (int e = 0; e < EPOCH; e++) {
    size_t kernel_loop0_off[1] = { 0 };
    size_t kernel_loop0_idx[1] = { SIZE };
    brisbane_task task1;
    brisbane_task_create(&task1);
    brisbane_task_kernel(task1, kernel_loop0, 1, kernel_loop0_off, kernel_loop0_idx);
    brisbane_task_submit(task1, brisbane_profile, NULL, true);
    brisbane_task_release(task1);
    /*
#pragma brisbane kernel present(C[0:SIZE], A[0:SIZE], B[0:SIZE]) device(gpu)
for (int i = 0; i < SIZE; i++) {
C[i] += A[i] + B[i];
}
*/
  }

  brisbane_task task2;
  brisbane_task_create(&task2);
  brisbane_task_d2h_full(task2, mem_C, C);
  brisbane_task_submit(task2, brisbane_data, NULL, true);

  for (int i = 0; i < SIZE; i++) {
    printf("[%8d] %8d = (%8d + %8d) * %d\n", i, C[i], A[i], B[i], EPOCH);
    if (C[i] != (A[i] + B[i]) * EPOCH) ERROR++;
  }
  printf("ERROR[%d]\n", ERROR);

  brisbane_task_release(task0);
  brisbane_task_release(task2);
  brisbane_kernel_release(kernel_loop0);
  brisbane_mem_release(mem_A);
  brisbane_mem_release(mem_B);
  brisbane_mem_release(mem_C);

  free(A);
  free(B);
  free(C);

  brisbane_finalize();

  return 0;
}
