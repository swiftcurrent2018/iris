#!/usr/bin/env python

import brisbane

brisbane.init(True)

task4 = brisbane.task_create("A")
brisbane.task_submit(task4, brisbane.brisbane_cpu, False);

task5 = brisbane.task_create("B")
brisbane.task_submit(task5, brisbane.brisbane_gpu, False);

task6 = brisbane.task_create("C")
task6_dep = [ task5 ]
brisbane.task_depend(task6, 1, task6_dep);
brisbane.task_submit(task6, brisbane.brisbane_cpu, False);

task7 = brisbane.task_create("D")
task7_dep = [ task4, task6 ]
brisbane.task_depend(task7, 2, task7_dep);
brisbane.task_submit(task7, brisbane.brisbane_gpu, False);

task8 = brisbane.task_create("E")
task8_dep = [ task5 ]
brisbane.task_depend(task8, 1, task8_dep);
brisbane.task_submit(task8, brisbane.brisbane_cpu, False);

brisbane.finalize()

