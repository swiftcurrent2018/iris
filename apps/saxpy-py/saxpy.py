#!/usr/bin/env python

import brisbane
import numpy as np
import sys

brisbane.init()

#print 'nplatforms:', brisbane.info_nplatforms()
#print 'ndevs:', brisbane.info_ndevs()

t1 = brisbane.timer_now()

SIZE = 8 if len(sys.argv) == 1 else int(sys.argv[1])
A = 10.0

x = np.arange(SIZE, dtype=np.float32)
y = np.arange(SIZE, dtype=np.float32)
z = np.arange(SIZE, dtype=np.float32)

print 'X', x
print 'Y', y

mem_x = brisbane.mem(x.nbytes)
mem_y = brisbane.mem(y.nbytes)
mem_z = brisbane.mem(z.nbytes)

kernel0 = brisbane.kernel("saxpy0")
kernel0.setmem(0, mem_z, brisbane.brisbane_w)
kernel0.setint(1, A)
kernel0.setmem(2, mem_x, brisbane.brisbane_r)

off = [ 0 ]
ndr = [ SIZE ]

task0 = brisbane.task()
task0.h2d_full(mem_x, x)
task0.h2d_full(mem_y, y)
task0.kernel(kernel0, 1, off, ndr)
task0.submit(brisbane.brisbane_gpu)

kernel1 = brisbane.kernel("saxpy1")
kernel1.setmem(0, mem_z, brisbane.brisbane_rw)
kernel1.setmem(1, mem_y, brisbane.brisbane_r)

task1 = brisbane.task()
task1.kernel(kernel1, 1, off, ndr)
task1.d2h_full(mem_z, z)
task1.submit(brisbane.brisbane_data)

t2 = brisbane.timer_now()

print 'Z =', A, '* X + Y', z
#print "execution time:", t2 - t1, "secs"

brisbane.finalize()

