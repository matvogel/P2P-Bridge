import os
from typing import Callable, Optional

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from .utils import *


class NPZFolderTest(Dataset):
    def __init__(self, root: str, features: Optional[str] = None):
        """
        Initialize the NPZFolderTest dataset.

        Args:
            root (str): Path to the root directory.
            features (Optional[str]): Features to include in the dataset.
        """
        super().__init__()
        self.root = root
        self.features = features
        self.files = load_npz_folder(root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = self.files[index]
        points = data["points"]
        features = data[self.features] if self.features is not None else None

        # normalize the points
        center = np.mean(points, axis=0)
        points -= center
        scale = np.max(np.linalg.norm(points, axis=1))
        points /= scale

        data = {
            "idx": index,
            "train_points": torch.from_numpy(points).float(),
            "train_points_center": center,
            "train_points_scale": scale,
        }

        if features is not None:
            data["features"] = torch.from_numpy(features).float()

        return data


class ScanNetPP_NPZ(Dataset):
    def __init__(
        self,
        root: str,
        mode: str = "training",
        additional_features: bool = False,
        augment: bool = False,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the ScanNetPP_NPZ dataset.

        Args:
            root (str): Path to the root directory.
            mode (str): Mode of the dataset (training or validation).
            additional_features (bool): Whether to include additional features.
            augment (bool): Whether to apply augmentation.
            transform (Optional[Callable]): Transform to apply to the data.
        """
        super().__init__()
        self.root = root
        self.mode = mode
        self.additional_features = additional_features
        self.augment = augment if mode == "training" else False
        self.transform = transform

        splits_path = "splits"
        with open(os.path.join(splits_path, "snpp_train.txt"), "r") as f:
            train_scans = f.read().splitlines()
        with open(os.path.join(splits_path, "snpp_val.txt"), "r") as f:
            val_scans = f.read().splitlines()

        # setup the splits
        if mode == "training":
            scans = train_scans
        elif mode == "validation":
            scans = val_scans
        else:
            raise NotImplementedError(f"Mode {mode} not implemented!")

        # scan paths for ply files
        folders = os.listdir(self.root)
        logger.info(f"Setting up preprocessed {mode} scannet dataset")
        folders = [f for f in folders if os.path.isdir(os.path.join(self.root, f))]
        folders = [f for f in folders if f in scans]

        self.scene_batches = []

        for folder in folders:
            folder_files = os.listdir(os.path.join(self.root, folder))
            points_paths = sorted([f for f in folder_files if f.startswith("points") and f.endswith(".npz")])
            for points in points_paths:
                data = {
                    "scene": folder,
                    "npz": os.path.join(self.root, folder, points),
                }
                self.scene_batches.append(data)

        logger.info(f"Loaded {len(self.scene_batches)} batches")

    def __len__(self):
        return len(self.scene_batches)


class ScanNetPP(ScanNetPP_NPZ):
    def __init__(
        self,
        root: str,
        mode: str = "training",
        additional_features: bool = False,
        augment: bool = False,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the ScanNetPP dataset.

        Args:
            root (str): Path to the root directory.
            mode (str): Mode of the dataset (training or validation).
            additional_features (bool): Whether to include additional features.
            augment (bool): Whether to apply augmentation.
            transform (Optional[Callable]): Transform to apply to the data.
        """
        super().__init__(
            root=root, mode=mode, additional_features=additional_features, augment=augment, transform=transform
        )

    def __getitem__(self, index):
        batch_data = {}
        while True:
            try:
                data = self.scene_batches[index]
                data_dict = np.load(data["npz"])
                clean = data_dict["clean"]
                noisy = data_dict["noisy"]
                break
            except Exception as e:
                logger.error(f"Failed to load data {data}")
                logger.exception(e)
                index = np.random.randint(0, self.__len__())

        # extract the points
        points_noisy = noisy[:, :3]
        points_clean = clean[:, :3]

        # extract the colors
        if noisy.shape[1] > 3:
            batch_data["noisy_colors"] = torch.from_numpy(noisy[:, 3:]).float()
        if clean.shape[1] > 3:
            batch_data["clean_colors"] = torch.from_numpy(clean[:, 3:]).float()

        # append the features if they are available
        if self.additional_features:
            features = data_dict["features"]
            batch_data["noisy_features"] = torch.from_numpy(features).float()

        # normalize the point coordinates
        if "center" not in data_dict:
            center = np.mean(points_noisy, axis=0)
            points_noisy -= center
            points_clean -= center
        else:
            center = data_dict["center"]

        if "scale" not in data_dict:
            scale = np.max(np.linalg.norm(points_noisy, axis=1))
            points_noisy /= scale
            points_clean /= scale
        else:
            scale = data_dict["scale"]

        # random rotation augmentation
        if self.augment and np.random.rand() < 0.5:
            points_noisy, theta = random_rotate_pointcloud_horizontally(points_noisy)
            points_clean, theta = random_rotate_pointcloud_horizontally(points_clean, theta=theta)

        # shuffle the point indexes
        rand_idxs = np.arange(points_noisy.shape[0])
        np.random.shuffle(rand_idxs)

        points_noisy = points_noisy[rand_idxs]
        points_clean = points_clean[rand_idxs]
        if "noisy_colors" in batch_data:
            batch_data["noisy_colors"] = batch_data["noisy_colors"][rand_idxs]
        if "clean_colors" in batch_data:
            batch_data["clean_colors"] = batch_data["clean_colors"][rand_idxs]
        if "noisy_features" in batch_data:
            batch_data["noisy_features"] = batch_data["noisy_features"][rand_idxs]

        if self.transform is not None:
            points_noisy = self.transform(points_noisy)
            points_clean = self.transform(points_clean)

        batch_data["idx"] = index
        batch_data["noisy_points"] = torch.from_numpy(points_clean).float()
        batch_data["clean_points"] = torch.from_numpy(points_noisy).float()
        batch_data["center"] = center
        batch_data["scale"] = scale

        return batch_data
