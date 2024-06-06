Here, we learn to write cuda kernels with a tight feedback loop, by using numba cuda, profiling with pytorch, and profiling with ncu.

**Result:** The matmuls written in numba-cuda are ~2-10x slower than the cuda-c kernels (from [Simon Boehm's blog post](https://siboehm.com/articles/22/CUDA-MMM)). The reason seems to be that numba-cuda adds memory safety checks. See https://numba.discourse.group/t/numba-cuda-slower-than-cuda-c-turn-off-memory-safety-checks/2554.
