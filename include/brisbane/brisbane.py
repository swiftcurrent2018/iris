import ctypes
from ctypes import *

dll_cl = CDLL("libOpenCL.so",   mode=RTLD_GLOBAL)
dll    = CDLL("libbrisbane.so", mode=RTLD_GLOBAL)

brisbane_r          =   (1 << 0)
brisbane_w          =   (1 << 1)
brisbane_rw         =   (brisbane_r | brisbane_w)

brisbane_default    =   (1 << 0)
brisbane_cpu        =   (1 << 1)
brisbane_nvidia     =   (1 << 2)
brisbane_amd        =   (1 << 3)
brisbane_gpu        =   (brisbane_nvidia | brisbane_amd)
brisbane_phi        =   (1 << 4)
brisbane_fpga       =   (1 << 5)
brisbane_data       =   (1 << 6)
brisbane_profile    =   (1 << 7)
brisbane_eager      =   (1 << 8)
brisbane_random     =   (1 << 9)
brisbane_any        =   (1 << 10)
brisbane_all        =   (1 << 11)

class brisbane_kernel(Structure):
    _fields_ = [("class_obj", c_void_p)]

class brisbane_mem(Structure):
    _fields_ = [("class_obj", c_void_p)]

class brisbane_task(Structure):
    _fields_ = [("class_obj", c_void_p)]

def init():
    return dll.brisbane_init(0, None)

def finalize():
    return dll.brisbane_finalize()

def synchronize():
    return dll.brisbane_synchronize()

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

def task_create():
    t = brisbane_task()
    dll.brisbane_task_create(byref(t))
    return t

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

def task_submit(task, device, wait):
    return dll.brisbane_task_submit(task, c_int(device), c_bool(wait))

def task_wait(task):
    return dll.brisbane_task_wait(task)

def task_add_subtask(task, subtask):
    return dll.brisbane_task_add_subtask(task, subtask)

def task_release(task):
    return dll.brisbane_task_release(task)

def task_release_mem(task, mem):
    return dll.brisbane_task_release_mem(task, mem)

