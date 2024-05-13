Here, we learn to write cuda kernels with a tight feedback loop, by using numba cuda, profiling with pytorch, and profiling with ncu.

**Current status:** The matmuls written in numba-cuda are ~2-10x slower than the cuda-c kernels (from [Simon Boehm's blog post](https://siboehm.com/articles/22/CUDA-MMM)). The reason is not clear, but the ptx suggests (i) multiplaying fp32 values results in an fp64, which is then cast back to fp32, and (ii) the branching logic is in more complicated.
