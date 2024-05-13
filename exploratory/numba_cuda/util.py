from functools import singledispatch
from fastcore.basics import tuplify
from numba import cuda

@singledispatch
def cdiv(a, b):
    return (a + b - 1) // b

@cdiv.register
def _(a: tuple, b):
    b = tuplify(b)
    if len(b) == 1:
        b *= len(a)
    return tuple((x + y - 1) // y for x, y in zip(a, b))

assert cdiv(10,3)==4
assert cdiv((10,10), (3))==(4,4)
assert cdiv((10,10), (3,4))==(4,3)

to_d = lambda o: cuda.to_device(o) # to device
to_h = lambda o: o.copy_to_host()  # to host
array_like = cuda.device_array_like
