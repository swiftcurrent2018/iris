#!/usr/bin/env python

import brisbane

brisbane.init()

task2 = brisbane.task_create()
brisbane.task_submit(task2, brisbane.brisbane_cpu, False);

task3 = brisbane.task_create()
brisbane.task_submit(task3, brisbane.brisbane_gpu, False);

task4 = brisbane.task_create()
task4_dep = [ task3 ]
brisbane.task_depend(task4, 1, task4_dep);
brisbane.task_submit(task4, brisbane.brisbane_cpu, False);

task5 = brisbane.task_create()
task5_dep = [ task2, task4 ]
brisbane.task_depend(task5, 2, task5_dep);
brisbane.task_submit(task5, brisbane.brisbane_gpu, False);

task6 = brisbane.task_create()
task6_dep = [ task2 ]
brisbane.task_depend(task6, 1, task6_dep);
brisbane.task_submit(task6, brisbane.brisbane_cpu, False);

brisbane.finalize()

