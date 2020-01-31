#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  size_t SIZE;
  int *SRC, *SINK;

  brisbane_init(&argc, &argv, 1);

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  printf("SIZE[%lu]\n", SIZE);

  int* AB = (int*) malloc(SIZE * sizeof(int));
  int* BC = (int*) malloc(SIZE * sizeof(int));

  brisbane_mem mem_AB;
  brisbane_mem mem_BC;

  brisbane_mem_create(SIZE * sizeof(int), &mem_AB);
  brisbane_mem_create(SIZE * sizeof(int), &mem_BC);

  brisbane_kernel kernel_A;
  brisbane_kernel_create("kernel_A", &kernel_A);
  brisbane_kernel_setmem(kernel_A, 0, mem_AB, brisbane_w);

  brisbane_kernel kernel_B;
  brisbane_kernel_create("kernel_B", &kernel_B);
  brisbane_kernel_setmem(kernel_B, 0, mem_AB, brisbane_r);
  brisbane_kernel_setmem(kernel_B, 1, mem_BC, brisbane_w);

  brisbane_kernel kernel_C;
  brisbane_kernel_create("kernel_C", &kernel_C);
  brisbane_kernel_setmem(kernel_C, 0, mem_BC, brisbane_r);

  brisbane_task task_A;
  brisbane_task_create(&task_A);
  brisbane_task_h2d_full(task_A, mem_AB, AB);
  brisbane_task_kernel(task_A, kernel_A, 1, NULL, &SIZE);

  brisbane_task task_B;
  brisbane_task_create(&task_B);
  brisbane_task_kernel(task_B, kernel_B, 1, NULL, &SIZE);
  brisbane_task_depend(task_B, 1, &task_A);

  brisbane_task task_C;
  brisbane_task_create(&task_C);
  brisbane_task_kernel(task_C, kernel_C, 1, NULL, &SIZE);
  brisbane_task_d2h_full(task_C, mem_BC, BC);
  brisbane_task_depend(task_C, 1, &task_B);

  brisbane_task_submit(task_A, brisbane_cpu, NULL, false);
  brisbane_task_submit(task_B, brisbane_cpu, NULL, false);
  brisbane_task_submit(task_C, brisbane_cpu, NULL, false);

  brisbane_synchronize();

  for (int i = 0; i < SIZE; i++) {
    printf("[%3d] %10d\n", i, BC[i]);
  }

  brisbane_finalize();

  return 0;
}
