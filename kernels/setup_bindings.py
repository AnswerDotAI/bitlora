from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="qdora",
    ext_modules=[
        CUDAExtension("qdora", ["pytorch_bindings.cuh", "kernels.cuh"]),
    ],
    cmdclass={"build_ext": BuildExtension},
)
