#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  brisbane_init(&argc, &argv, true);

  brisbane_policy_register("policy_last.so", "policy_last");

  brisbane_task task;
  brisbane_task_create(&task);
  brisbane_task_submit(task, brisbane_custom, "policy_last", false);

  brisbane_finalize();

  return 0;
}

