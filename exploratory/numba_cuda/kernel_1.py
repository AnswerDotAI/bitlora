# a naive matmul implementation, to test profiling
import numpy as np
from fastcore.basics import tuplify

from numba import cuda
from util import to_d, to_h, array_like, cdiv

m,n,k = 4092,4092,4092

a = to_d(np.ones((m,k)))
b = to_d(np.ones((k,n)))
c = to_d(np.empty((m,n)))

@cuda.jit(lineinfo=True)
def matmul_1(a,b,c):
    tid_x, tid_y = cuda.grid(2)
    m,n = c.shape
    k = a.shape[1]
    if tid_x >= m or tid_y >= n: return 
    tmp = 0
    for i in range(k):
        tmp += a[tid_x, i] * b[i, tid_y]        
    c[tid_x, tid_y] = tmp

nthreads = (32,32)
nblocks = cdiv(c.shape, nthreads)

matmul_1[nblocks, nthreads](a,b,c)
