from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="pointnet2_cuda",
    ext_modules=[
        CUDAExtension(
            "pointnet2_batch_cuda",
            [
                "src/pointnet2_api.cpp",
                "src/ball_query.cpp",
                "src/ball_query_gpu.cu",
                "src/group_points.cpp",
                "src/group_points_gpu.cu",
                "src/interpolate.cpp",
                "src/interpolate_gpu.cu",
                "src/sampling.cpp",
                "src/sampling_gpu.cu",
                "src/vox.cpp",
                "src/vox_gpu.cu",
                "src/trilinear_devox.cpp",
                "src/trilinear_devox_gpu.cu",
                "src/pvcnn_ball_query.cpp",
                "src/pvcnn_ball_query_gpu.cu",
                "src/pvcnn_neighbor_interpolate_gpu.cu",
                "src/pvcnn_neighbor_interpolate.cpp",
                "src/pvcnn_grouping.cpp",
                "src/pvcnn_grouping_gpu.cu",
                "src/pvcnn_sampling.cpp",
                "src/pvcnn_sampling_gpu.cu",
            ],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    # add extensions
    # ext_modules=[
)
