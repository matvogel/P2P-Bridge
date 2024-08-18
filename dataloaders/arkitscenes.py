import os
from typing import List, Optional

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from dataloaders.utils import random_rotate_pointcloud_horizontally


class ArkitNPZ(Dataset):
    """Arkit dataset class for loading preprocessed npz files."""

    def __init__(
        self, root: str, mode: str = "training", features: Optional[str] = None, augment: Optional[str] = None
    ):
        """
        Initialize the Arkit dataset.

        Args:
            root (str): Path to the root directory.
            mode (str): Mode of the dataset (training or validation).
            features (Optional[str]): Features to include in the dataset.
            augment (Optional[str]): Augmentation to apply to the dataset.
        """
        super().__init__()
        self.mode = mode
        self.features = features
        self.augment = augment if mode == "training" else False

        # scan paths for ply files
        logger.info(f"Setting up preprocessed {mode} ArKit dataset")
        data_root = os.path.join(root, "train" if mode == "training" else "val")
        self.root = data_root
        room_folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]

        self.scene_batches = []

        for folder in room_folders:
            visit_ids = os.listdir(os.path.join(self.root, folder))
            for visit_id in visit_ids:
                folder_files = os.listdir(os.path.join(self.root, folder, visit_id))
                points_paths = sorted([f for f in folder_files if f.startswith("points") and f.endswith(".npz")])
                for points in points_paths:
                    data = {
                        "room_id": folder,
                        "visit_id": visit_id,
                        "npz": os.path.join(self.root, folder, visit_id, points),
                    }
                    self.scene_batches.append(data)

        logger.info(f"Loaded {len(self.scene_batches)} batches")

    def __len__(self):
        return len(self.scene_batches)

    def __getitem__(self, index):
        batch_data = {}

        data = self.scene_batches[index % len(self.scene_batches)]
        data_dict = np.load(data["npz"])

        faro = data_dict["faro"]
        iphone = data_dict["iphone"]

        # extract the points
        points_iphone = iphone[:, :3]
        points_faro = faro[:, :3]

        # extract the colors
        if iphone.shape[1] > 3:
            batch_data["lr_colors"] = torch.from_numpy(iphone[:, 3:]).float()
        if faro.shape[1] > 3:
            batch_data["hr_colors"] = torch.from_numpy(faro[:, 3:]).float()

        # append the features if they are available
        if self.features is not None:
            features = data_dict[self.features]
            batch_data["lr_features"] = torch.from_numpy(features).float()

        # normalize the point coordinates
        if "center" in batch_data:
            center = batch_data["center"]
        else:
            center = np.mean(points_iphone, axis=0)
            points_iphone -= center
            points_faro -= center

        if "scale" in batch_data:
            scale = batch_data["scale"]
        else:
            scale = np.max(np.linalg.norm(points_iphone, axis=1))
            points_iphone /= scale
            points_faro /= scale

        # random rotation augmentation
        if self.augment and np.random.rand() < 0.5:
            points_iphone, theta = random_rotate_pointcloud_horizontally(points_iphone)
            points_faro, theta = random_rotate_pointcloud_horizontally(points_faro, theta=theta)

        batch_data["idx"] = index
        batch_data["hr_points"] = torch.from_numpy(points_faro).float()
        batch_data["lr_points"] = torch.from_numpy(points_iphone).float()
        batch_data["center"] = center
        batch_data["scale"] = scale

        return batch_data
