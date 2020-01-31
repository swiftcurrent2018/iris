#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char** argv) {
  size_t SIZE;
  int LOOP;
  double t0, t1;

  brisbane_init(&argc, &argv, 1);

  SIZE = argc > 1 ? atol(argv[1]) : 128;
  LOOP = argc > 2 ? atoi(argv[2]) : 10000;
  printf("SIZE[%lu] LOOP[%d]\n", SIZE, LOOP);

  brisbane_mem mem0, mem1, mem2, mem3;
  brisbane_mem_create(SIZE * sizeof(int), &mem0);
  brisbane_mem_create(SIZE * sizeof(int), &mem1);
  brisbane_mem_create(SIZE * sizeof(int), &mem2);
  brisbane_mem_create(SIZE * sizeof(int), &mem3);

  brisbane_kernel kernel0, kernel1, kernel2, kernel3;
  brisbane_kernel_create("kernel0", &kernel0);
  brisbane_kernel_create("kernel1", &kernel1);
  brisbane_kernel_create("kernel2", &kernel2);
  brisbane_kernel_create("kernel3", &kernel3);

  brisbane_kernel_setmem(kernel0, 0, mem0, brisbane_w);
  brisbane_kernel_setmem(kernel1, 0, mem1, brisbane_w);
  brisbane_kernel_setmem(kernel2, 0, mem2, brisbane_w);
  brisbane_kernel_setmem(kernel3, 0, mem3, brisbane_w);

  brisbane_kernel_setarg(kernel0, 1, sizeof(int), &LOOP);
  brisbane_kernel_setarg(kernel1, 1, sizeof(int), &LOOP);
  brisbane_kernel_setarg(kernel2, 1, sizeof(int), &LOOP);
  brisbane_kernel_setarg(kernel3, 1, sizeof(int), &LOOP);

  brisbane_task task0, task1, task2, task3;
  brisbane_task_create(&task0);
  brisbane_task_create(&task1);
  brisbane_task_create(&task2);
  brisbane_task_create(&task3);
  brisbane_task_kernel(task0, kernel0, 1, NULL, &SIZE);
  brisbane_task_kernel(task1, kernel1, 1, NULL, &SIZE);
  brisbane_task_kernel(task2, kernel2, 1, NULL, &SIZE);
  brisbane_task_kernel(task3, kernel3, 1, NULL, &SIZE);

  brisbane_task_submit(task0, 0, NULL, false);
  brisbane_task_submit(task1, 0, NULL, false);
  brisbane_task_submit(task2, 0, NULL, false);
  brisbane_task_submit(task3, 0, NULL, false);

  brisbane_synchronize();

  brisbane_finalize();

  return 0;
}
