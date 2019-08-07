#!/usr/bin/env python

import brisbane
import numpy as np
import sys

brisbane.init()

SIZE = 8 if len(sys.argv) == 1 else int(sys.argv[1])
A = 5.0

x = np.arange(SIZE, dtype=np.float32)
y = np.arange(SIZE, dtype=np.float32)
z = np.arange(SIZE, dtype=np.float32)

print 'X', x
print 'Y', y

mem_x = brisbane.mem_create(SIZE * 4)
mem_y = brisbane.mem_create(SIZE * 4)
mem_z = brisbane.mem_create(SIZE * 4)

kernel = brisbane.kernel_create("saxpy")
brisbane.kernel_setmem(kernel, 0, mem_z, brisbane.brisbane_w)
brisbane.kernel_setarg(kernel, 1, 4, A)
brisbane.kernel_setmem(kernel, 2, mem_x, brisbane.brisbane_r)
brisbane.kernel_setmem(kernel, 3, mem_y, brisbane.brisbane_r)

off = [ 0 ]
ndr = [ SIZE ]
task = brisbane.task_create()
brisbane.task_h2d_full(task, mem_x, x)
brisbane.task_h2d_full(task, mem_y, y)
brisbane.task_kernel(task, kernel, 1, off, ndr)
brisbane.task_d2h_full(task, mem_z, z)
brisbane.task_submit(task, brisbane.brisbane_eager, True)

print 'Z =', A, '* X + Y', z

brisbane.finalize()

