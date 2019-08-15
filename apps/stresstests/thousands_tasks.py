#!/usr/bin/env python

import brisbane
import numpy as np
import sys

brisbane.init()

NTASKS = 1024 if len(sys.argv) == 1 else int(sys.argv[1])

print 'NTASKS', NTASKS

for i in range(NTASKS):
  task = brisbane.task_create()
  brisbane.task_submit(task, brisbane.brisbane_any, False)

brisbane.finalize()

