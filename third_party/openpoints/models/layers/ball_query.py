from third_party.openpoints.cpp import pointnet2_cuda


def ball_query(centers_coords, points_coords, radius, num_neighbors):
    """
    :param centers_coords: coordinates of centers, FloatTensor[B, 3, M]
    :param points_coords: coordinates of points, FloatTensor[B, 3, N]
    :param radius: float, radius of ball query
    :param num_neighbors: int, maximum number of neighbors
    :return:
        neighbor_indices: indices of neighbors, IntTensor[B, M, U]
    """
    centers_coords = centers_coords[:, :3].contiguous()
    points_coords = points_coords[:, :3].contiguous()
    return pointnet2_cuda.ball_query(centers_coords, points_coords, radius, num_neighbors)
