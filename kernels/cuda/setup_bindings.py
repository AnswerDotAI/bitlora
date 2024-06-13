from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="qdora",
    ext_modules=[
        CUDAExtension("qdora", ["pytorch_bindings.cu"]),
    ],
    cmdclass={"build_ext": BuildExtension},
)
