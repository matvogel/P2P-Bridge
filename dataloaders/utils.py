import os

import numpy as np
from numpy.typing import ArrayLike


def random_rotate_pointcloud_horizontally(pointcloud: ArrayLike, theta: float = None):
    """Randomly rotate a point cloud horizontally.

    Args:
        pointcloud (ArrayLike): The point cloud to rotate.
        theta (float): The angle to rotate the point cloud by.

    Returns:
        ArrayLike: The rotated point cloud.
    """
    rotated = False
    if pointcloud.shape[-1] != 3:
        pointcloud = pointcloud.T
        rotated = True

    if theta is None:
        theta = np.random.rand() * 2 * np.pi

    cosval = np.cos(theta)
    sinval = np.sin(theta)
    rotation_matrix = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])

    rotated_pointcloud = np.dot(pointcloud, rotation_matrix)

    if rotated:
        rotated_pointcloud = rotated_pointcloud.T

    return rotated_pointcloud, theta


def load_npz_folder(folder: str):
    """Load a folder of .npz files.

    Args:
        folder (str): The path to the folder.
    """
    data = []
    for file in os.listdir(folder):
        if file.endswith(".npz"):
            data.append(load_npz(os.path.join(folder, file)))
    return data


def load_npz(path: str):
    """Load a .npz file.

    Args:
        path (str): The path to the .npz file.
    """
    data = np.load(path)
    return data
