import ctypes
from ctypes import *

dll = CDLL("libbrisbane.so", mode=RTLD_GLOBAL)

brisbane_default    =   (1 << 5)
brisbane_cpu        =   (1 << 6)
brisbane_nvidia     =   (1 << 7)
brisbane_amd        =   (1 << 8)
brisbane_gpu        =   (brisbane_nvidia | brisbane_amd)
brisbane_phi        =   (1 << 9)
brisbane_fpga       =   (1 << 10)
brisbane_data       =   (1 << 11)
brisbane_profile    =   (1 << 12)
brisbane_random     =   (1 << 13)
brisbane_any        =   (1 << 14)
brisbane_all        =   (1 << 15)

brisbane_r          =   (1 << 0)
brisbane_w          =   (1 << 1)
brisbane_rw         =   (brisbane_r | brisbane_w)

brisbane_int        =   (1 << 0)
brisbane_long       =   (1 << 1)
brisbane_float      =   (1 << 2)
brisbane_double     =   (1 << 3)

brisbane_normal     =   (1 << 10)
brisbane_reduction  =   (1 << 11)
brisbane_sum        =   ((1 << 12) | brisbane_reduction)
brisbane_max        =   ((1 << 13) | brisbane_reduction)
brisbane_min        =   ((1 << 14) | brisbane_reduction)

brisbane_platform   =   0x1001
brisbane_vendor     =   0x1002
brisbane_name       =   0x1003
brisbane_type       =   0x1004

class brisbane_kernel(Structure):
    _fields_ = [("class_obj", c_void_p)]

class brisbane_mem(Structure):
    _fields_ = [("class_obj", c_void_p)]

class brisbane_task(Structure):
    _fields_ = [("class_obj", c_void_p)]

def init(sync = 1):
    return dll.brisbane_init(0, None, c_int(sync))

def finalize():
    return dll.brisbane_finalize()

def synchronize():
    return dll.brisbane_synchronize()

def platform_count():
    i = c_int()
    dll.brisbane_platform_count(byref(i))
    return i.value

def platform_info(platform, param):
    s = (c_char * 64)()
    dll.brisbane_platform_info(c_int(platform), c_int(param), s, None)
    return s.value

def device_count():
    i = c_int()
    dll.brisbane_device_count(byref(i))
    return i.value

def device_info(device, param):
    s = (c_char * 64)()
    dll.brisbane_device_info(c_int(device), c_int(param), s, None)
    return s.value

def mem_create(size):
    m = brisbane_mem()
    dll.brisbane_mem_create(c_size_t(size), byref(m))
    return m

def mem_reduce(mem, mode, type):
    return dll.brisbane_mem_reduce(mem, c_int(mode), c_int(type))

def mem_release(mem):
    return dll.brisbane_mem_release(mem)

def kernel_create(name):
    k = brisbane_kernel()
    dll.brisbane_kernel_create(c_char_p(name), byref(k))
    return k

def kernel_setarg(kernel, idx, size, value):
    if type(value) == int: cvalue = byref(c_int(value))
    elif type(value) == float and size == 4: cvalue = byref(c_float(value))
    elif type(value) == float and size == 8: cvalue = byref(c_double(value))
    return dll.brisbane_kernel_setarg(kernel, c_int(idx), c_size_t(size), cvalue)

def kernel_setmem(kernel, idx, mem, mode):
    return dll.brisbane_kernel_setmem(kernel, c_int(idx), mem, c_int(mode))

def kernel_release(kernel):
    return dll.brisbane_kernel_release(kernel)

def task_create(name = None):
    t = brisbane_task()
    dll.brisbane_task_create_name(c_char_p(name), byref(t))
    return t

def task_depend(task, ntasks, tasks):
    return dll.brisbane_task_depend(task, c_int(ntasks), (brisbane_task * len(tasks))(*tasks))

def task_kernel(task, kernel, dim, off, ndr):
    coff = (c_size_t * dim)(*off)
    cndr = (c_size_t * dim)(*ndr)
    return dll.brisbane_task_kernel(task, kernel, c_int(dim), coff, cndr)

def task_h2d(task, mem, off, size, host):
    return dll.brisbane_task_h2d(task, mem, c_size_t(off), c_size_t(size), host.ctypes.data_as(c_void_p))

def task_d2h(task, mem, off, size, host):
    return dll.brisbane_task_d2h(task, mem, c_size_t(off), c_size_t(size), host.ctypes.data_as(c_void_p))

def task_h2d_full(task, mem, host):
    return dll.brisbane_task_h2d_full(task, mem, host.ctypes.data_as(c_void_p))

def task_d2h_full(task, mem, host):
    return dll.brisbane_task_d2h_full(task, mem, host.ctypes.data_as(c_void_p))

def task_submit(task, device, sync):
    return dll.brisbane_task_submit(task, c_int(device), None, c_int(sync))

def task_wait(task):
    return dll.brisbane_task_wait(task)

def task_wait_all(ntask, tasks):
    return dll.brisbane_task_wait_all(c_int(ntask), (brisbane_task * len(tasks))(*tasks))

def task_add_subtask(task, subtask):
    return dll.brisbane_task_add_subtask(task, subtask)

def task_release(task):
    return dll.brisbane_task_release(task)

def task_release_mem(task, mem):
    return dll.brisbane_task_release_mem(task, mem)

def timer_now():
    d = c_double()
    dll.brisbane_timer_now(byref(d))
    return d.value

class mem:
  def __init__(self, size):
    self.handle = mem_create(size)
  def reduce(self, mode, type):
    mem_reduce(self.handle, mode, type)
  def release(self):
    mem_release(self.handle)

class kernel:
  def __init__(self, name):
    self.handle = kernel_create(name)
  def setarg(self, idx, size, value):
    kernel_setarg(self.handle, idx, size, value)
  def setint(self, idx, value):
    kernel_setarg(self.handle, idx, 4, value)
  def setfloat(self, idx, value):
    kernel_setarg(self.handle, idx, 8, value)
  def setdouble(self, idx, value):
    kernel_setarg(self.handle, idx, 8, value)
  def setmem(self, idx, mem, mode):
    kernel_setmem(self.handle, idx, mem.handle, mode)
  def release(self):
    kernel_release(self.handle)

class task:
  def __init__(self):
    self.handle = task_create()
  def h2d(self, mem, off, size, host):
    task_h2d(self.handle, mem.handle, off, size, host)
  def d2h(self, mem, off, size, host):
    task_d2h(self.handle, meml.handle, off, size, host)
  def h2d_full(self, mem, host):
    task_h2d_full(self.handle, mem.handle, host)
  def d2h_full(self, mem, host):
    task_d2h_full(self.handle, mem.handle, host)
  def kernel(self, kernel, dim, off, ndr):
    task_kernel(self.handle, kernel.handle, dim, off, ndr)
  def submit(self, device, sync = 1):
    task_submit(self.handle, device, sync)

