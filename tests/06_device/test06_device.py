#!/usr/bin/env python

import brisbane

brisbane.init(True)

nplatforms = brisbane.platform_count()
for i in range(nplatforms):
  name = brisbane.platform_info(i, brisbane.brisbane_name)
  print "platform[", i, "] name[", name, "]"

ndevs = brisbane.device_count()
for i in range(ndevs):
  vendor = brisbane.device_info(i, brisbane.brisbane_vendor)
  name = brisbane.device_info(i, brisbane.brisbane_name)
  print "device[", i, "] vendor[", vendor, "] name[", name, "]"

brisbane.finalize()

