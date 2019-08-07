#!/usr/bin/env python

import brisbane
import numpy as np
import sys

brisbane.init()

SIZE = 8 if len(sys.argv) == 1 else int(sys.argv[1])

x = np.arange(SIZE * SIZE, dtype=np.float64).reshape((SIZE, SIZE))
y = np.arange(SIZE * SIZE, dtype=np.float64).reshape((SIZE, SIZE))
z = np.arange(SIZE * SIZE, dtype=np.float64).reshape((SIZE, SIZE))

print 'X[', SIZE, ',', SIZE, ']\n', x
print 'Y[', SIZE, ',', SIZE, ']\n', y

mem_x = brisbane.mem_create(SIZE * SIZE * 8)
mem_y = brisbane.mem_create(SIZE * SIZE * 8)
mem_z = brisbane.mem_create(SIZE * SIZE * 8)

kernel = brisbane.kernel_create("ijk")
brisbane.kernel_setmem(kernel, 0, mem_z, brisbane.brisbane_w)
brisbane.kernel_setmem(kernel, 1, mem_x, brisbane.brisbane_r)
brisbane.kernel_setmem(kernel, 2, mem_y, brisbane.brisbane_r)
brisbane.kernel_setarg(kernel, 3, 4, SIZE)

off = [ 0, 0 ]
ndr = [ SIZE, SIZE ]
task = brisbane.task_create()
brisbane.task_h2d_full(task, mem_x, x)
brisbane.task_h2d_full(task, mem_y, y)
brisbane.task_kernel(task, kernel, 2, off, ndr)
brisbane.task_d2h_full(task, mem_z, z)
brisbane.task_submit(task, brisbane.brisbane_eager, True)

print 'Z[', SIZE, ',', SIZE, '] = X * Y\n', z

brisbane.finalize()

