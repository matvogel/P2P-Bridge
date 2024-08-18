import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

python_version = "{}.{}".format(sys.version_info.major, sys.version_info.minor)

try:
    lib_path = os.path.join(
        os.environ["CONDA_PREFIX"],
        f"lib/python{python_version}/site-packages/torch/lib",
    )
except KeyError:
    raise KeyError("Couldn't find $CONDA_PREFIX, make sure your environment is activated!")

if not os.path.exists(lib_path):
    raise FileNotFoundError("Couldn't find {lib_path}, ensure that torch is installed.")

emd_module = CUDAExtension(
    "emd_cuda",
    sources=["cuda/emd.cpp", "cuda/emd_kernel.cu"],
    extra_compile_args={"cxx": ["-O2", "-g"], "nvcc": ["-O2"]},
    extra_link_args=["-Wl,-rpath," + lib_path],
)

setup(
    name="emd",
    py_modules=["emd"],
    ext_modules=[emd_module],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
