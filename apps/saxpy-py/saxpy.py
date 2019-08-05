#!/usr/bin/env python

import numpy as np
import brisbane

brisbane.init()

x = np.arange(10, dtype=np.float32)
y = np.arange(10, dtype=np.float32)
z = np.arange(10, dtype=np.float32)

print x
print y

mem_x = brisbane.mem_create(40)
mem_y = brisbane.mem_create(40)
mem_z = brisbane.mem_create(40)

kernel = brisbane.kernel_create("saxpy")

brisbane.kernel_setmem(kernel, 0, mem_z, brisbane.brisbane_w)
brisbane.kernel_setarg(kernel, 1, 4, 5)
brisbane.kernel_setmem(kernel, 2, mem_x, brisbane.brisbane_r)
brisbane.kernel_setmem(kernel, 3, mem_y, brisbane.brisbane_r)

off = [ 0 ]
ndr = [ 10 ]
task = brisbane.task_create()
brisbane.task_h2d_full(task, mem_x, x)
brisbane.task_h2d_full(task, mem_y, y)
brisbane.task_kernel(task, kernel, 1, off, ndr)
brisbane.task_d2h_full(task, mem_z, z)
brisbane.task_submit(task, brisbane.brisbane_gpu, True)

print z

brisbane.finalize()
