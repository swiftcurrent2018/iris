#!/usr/bin/env python

import brisbane
import numpy as np
import sys

brisbane.init()

NTASKS = 1024 if len(sys.argv) == 1 else int(sys.argv[1])

print 'NTASKS', NTASKS

t0 = brisbane.timer_now()

for i in range(NTASKS):
  task = brisbane.task()
  task.submit(brisbane.brisbane_random, False)

brisbane.synchronize()

t1 = brisbane.timer_now()

print 'Time:', t1 - t0

brisbane.finalize()

