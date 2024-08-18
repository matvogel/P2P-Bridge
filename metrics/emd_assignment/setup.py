from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="emd_assignment",
    ext_modules=[
        CUDAExtension(
            "emd_assignment",
            [
                "emd_assignment/emd.cpp",
                "emd_assignment/emd_cuda.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
