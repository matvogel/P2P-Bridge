#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "ball_query_gpu.h"
#include "group_points_gpu.h"
#include "sampling_gpu.h"
#include "interpolate_gpu.h"
#include "vox.hpp"
#include "trilinear_devox.hpp"
#include "pvcnn_ball_query.hpp"
#include "pvcnn_neighbor_interpolate.hpp"
#include "pvcnn_grouping.hpp"
#include "pvcnn_sampling.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ball_query_wrapper", &ball_query_wrapper_fast, "ball_query_wrapper_fast");

    m.def("group_points_wrapper", &group_points_wrapper_fast, "group_points_wrapper_fast");
    m.def("group_points_grad_wrapper", &group_points_grad_wrapper_fast, "group_points_grad_wrapper_fast");

    m.def("gather_points_wrapper", &gather_points_wrapper_fast, "gather_points_wrapper_fast");
    m.def("gather_points_grad_wrapper", &gather_points_grad_wrapper_fast, "gather_points_grad_wrapper_fast");

    m.def("furthest_point_sampling_wrapper", &furthest_point_sampling_wrapper, "furthest_point_sampling_wrapper");

    m.def("three_nn_wrapper", &three_nn_wrapper_fast, "three_nn_wrapper_fast");
    m.def("three_interpolate_wrapper", &three_interpolate_wrapper_fast, "three_interpolate_wrapper_fast");
    m.def("three_interpolate_grad_wrapper", &three_interpolate_grad_wrapper_fast, "three_interpolate_grad_wrapper_fast");

    m.def("avg_voxelize_backward", &avg_voxelize_backward, "avg_voxelize_backward");
    m.def("avg_voxelize_forward", &avg_voxelize_forward, "avg_voxelize_forward");

    m.def("trilinear_devoxelize_forward", &trilinear_devoxelize_forward, "trilinear_devoxelize_forward");
    m.def("trilinear_devoxelize_backward", &trilinear_devoxelize_backward, "trilinear_devoxelize_backward");

    m.def("ball_query", &ball_query_forward, "ball_query_forward");

    m.def("three_nearest_neighbors_interpolate_forward", &three_nearest_neighbors_interpolate_forward, "three_nearest_neighbors_interpolate_forward");
    m.def("three_nearest_neighbors_interpolate_backward", &three_nearest_neighbors_interpolate_backward, "three_nearest_neighbors_interpolate_backward");

    m.def("grouping_forward", &grouping_forward, "grouping_forward");
    m.def("grouping_backward", &grouping_backward, "grouping_backward");

    m.def("gather_features_forward", &gather_features_forward, "gather_features_forward");
    m.def("gather_features_backward", &gather_features_backward, "gather_features_backward");
    m.def("furthest_point_sampling_forward", &furthest_point_sampling_forward, "furthest_point_sampling_forward");
}
