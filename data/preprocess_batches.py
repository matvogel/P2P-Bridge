import argparse
import os
from typing import List

import numpy as np
import open3d as o3d
import torch
from loguru import logger
from omegaconf import DictConfig
from processing.utils import create_spherical_batches
from torch.multiprocessing import spawn
from tqdm import tqdm


def handle_folder(idx: int, folder_batches: List, args: DictConfig) -> None:
    """
    Handle a list of folders.

    Args:
        idx (int): ID of the process.
        folder_batches (List): List of scene IDs to process.
        args (DictConfig): Configuration dictionary.
    """
    data_folders = folder_batches[idx]

    for data_folder in tqdm(data_folders, desc=f"Process {idx}"):
        faro_scan_path = os.path.join(args.data_root, data_folder, "scans", "mesh_aligned_0.05.ply")
        iphone_scan_path = os.path.join(args.data_root, data_folder, "scans", f"iphone{args.name_suffix}.ply")

        if not os.path.exists(faro_scan_path) or not os.path.exists(iphone_scan_path):
            logger.info("Skipping", data_folder, "because of missing scans")
            continue

        # feature paths creation and check
        if args.feature_type is not None:
            fpath = os.path.join(
                args.data_root, data_folder, "features", f"{args.feature_type}_iphone{args.name_suffix}.npy"
            )
            if os.path.exists(fpath):
                features = np.load(fpath).T
            else:
                logger.info("Skipping", data_folder, "because of missing features")
                continue
        else:
            features = None

        # target scene path
        target_scene_path = os.path.join(args.output_root, data_folder)
        os.makedirs(target_scene_path, exist_ok=True)

        # load iphone cloud and faro mesh and oversample faro to faciliate the nearest neighbor calculation
        scan_iphone = o3d.io.read_point_cloud(iphone_scan_path)
        mesh_faro = o3d.io.read_triangle_mesh(faro_scan_path)

        # extract points from iphone scan
        xyz_iphone = np.array(scan_iphone.points)
        rgb_iphone = np.array(scan_iphone.colors)

        # oversample
        scan_faro = mesh_faro.sample_points_uniformly(len(xyz_iphone) * 5)
        xyz_faro = np.array(scan_faro.points)
        rgb_faro = np.array(scan_faro.colors)

        n_points_iphone = xyz_iphone.shape[0]

        if features is not None and features.shape[0] != n_points_iphone:
            logger.info(
                "Scene {} has {} points but {} features".format(
                    data_folder,
                    n_points_iphone,
                    features.shape[0],
                )
            )
            continue

        # get batches
        batches = create_spherical_batches(
            pcd_clean=xyz_faro,
            pcd_noisy=xyz_iphone,
            rgb_clean=rgb_faro,
            rgb_noisy=rgb_iphone,
            features=features,
            args=args,
        )

        # save batches
        for batch_idx in range(len(batches)):
            np.savez(
                os.path.join(target_scene_path, "points_{}.npz".format(batch_idx)),
                **batches[batch_idx],
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--name_suffix",
        type=str,
        default="",
        help="Suffix to append to the name of the points and features.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Path to the target directory.",
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=4096,
        help="Number of batches per scene.",
    )
    parser.add_argument(
        "--r",
        type=float,
        default=0.3,
        help="Radius for the spherical batches.",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="dino",
        help="Features to use.",
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=4,
        help="Number of processes to spawn.",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    data_folders = os.listdir(args.data_root)
    data_folders = [f for f in data_folders if os.path.isdir(os.path.join(args.data_root, f))]

    # create batches of folders to split the work among processes
    n_batches = int(len(data_folders) / args.nprocs)
    batches = [data_folders[i : i + n_batches] for i in range(0, len(data_folders), n_batches)]

    spawn(handle_folder, args=(batches, args), nprocs=args.nprocs, join=True)


if __name__ == "__main__":
    main()
