#!/usr/bin/env python

import brisbane
import numpy as np
import sys

brisbane.init()

task2 = brisbane.task_create("A")
brisbane.task_submit(task2, brisbane.brisbane_any, False);

task3 = brisbane.task_create("B")
brisbane.task_submit(task3, brisbane.brisbane_any, False);

task4 = brisbane.task_create("C")
task4_dep = [ task3 ]
brisbane.task_depend(task4, 1, task4_dep);
brisbane.task_submit(task4, brisbane.brisbane_any, False);

task7 = brisbane.task_create("BLOCK")
brisbane.task_submit(task7, brisbane.brisbane_any, True);
brisbane.task_release(task7);

task8 = brisbane.task_create("NB")
brisbane.task_submit(task8, brisbane.brisbane_any, False);

task5 = brisbane.task_create("D")
task5_dep = [ task2, task4 ]
brisbane.task_depend(task5, 2, task5_dep);
brisbane.task_submit(task5, brisbane.brisbane_any, False);

task6 = brisbane.task_create("E")
task6_dep = [ task2 ]
brisbane.task_depend(task6, 1, task6_dep);
brisbane.task_submit(task6, brisbane.brisbane_any, False);

brisbane.finalize()

