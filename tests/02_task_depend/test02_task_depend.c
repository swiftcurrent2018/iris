#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  brisbane_init(&argc, &argv, true);

  brisbane_task task2;
  brisbane_task_create(&task2);
  brisbane_task_submit(task2, brisbane_cpu, NULL, false);

  brisbane_task task3;
  brisbane_task_create(&task3);
  brisbane_task_submit(task3, brisbane_gpu, NULL, false);

  brisbane_task task4;
  brisbane_task task4_dep[] = { task3 };
  brisbane_task_create(&task4);
  brisbane_task_depend(task4, 1, task4_dep);
  brisbane_task_submit(task4, brisbane_cpu, NULL, false);

  brisbane_task task5;
  brisbane_task task5_dep[] = { task2, task4 };
  brisbane_task_create(&task5);
  brisbane_task_depend(task5, 2, task5_dep);
  brisbane_task_submit(task5, brisbane_gpu, NULL, false);

  brisbane_task task6;
  brisbane_task task6_dep[] = { task2 };
  brisbane_task_create(&task6);
  brisbane_task_depend(task6, 1, task6_dep);
  brisbane_task_submit(task6, brisbane_cpu, NULL, false);

  brisbane_finalize();

  return 0;
}
