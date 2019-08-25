#!/usr/bin/env python

import brisbane
import numpy as np
import sys

brisbane.init(False)

#print 'nplatforms:', brisbane.info_nplatforms()
#print 'ndevs:', brisbane.info_ndevs()

t1 = brisbane.timer_now()

SIZE = 8 if len(sys.argv) == 1 else int(sys.argv[1])
A = 5.0

x = np.arange(SIZE, dtype=np.float32)
y = np.arange(SIZE, dtype=np.float32)
z = np.arange(SIZE, dtype=np.float32)

print 'X', x
print 'Y', y

mem_x = brisbane.mem(SIZE * 4)
mem_y = brisbane.mem(SIZE * 4)
mem_z = brisbane.mem(SIZE * 4)

kernel = brisbane.kernel("saxpy")
kernel.setmem(0, mem_z, brisbane.brisbane_w)
kernel.setarg(1, 4, A)
kernel.setmem(2, mem_x, brisbane.brisbane_r)
kernel.setmem(3, mem_y, brisbane.brisbane_r)

off = [ 0 ]
ndr = [ SIZE ]

task = brisbane.task()
task.h2d_full(mem_x, x)
task.h2d_full(mem_y, y)
task.kernel(kernel, 1, off, ndr)
task.d2h_full(mem_z, z)
task.submit(brisbane.brisbane_gpu, True)

t2 = brisbane.timer_now()

print 'Z =', A, '* X + Y', z
#print "execution time:", t2 - t1, "secs"

brisbane.finalize()

