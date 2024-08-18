import glob
import logging
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ...transforms.point_transform_cpu import PointsToTensor
from ..build import DATASETS
from ..data_util import crop_pc

VALID_CLASS_IDS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    16,
    24,
    28,
    33,
    34,
    36,
    39,
]

SCANNET_COLOR_MAP = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}


@DATASETS.register_module()
class ScanNet(Dataset):
    num_classes = 20
    classes = [
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refrigerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "otherfurniture",
    ]
    gravity_dim = 2

    color_mean = [0.46259782, 0.46253258, 0.46253258]
    color_std = [0.693565, 0.6852543, 0.68061745]
    """ScanNet dataset, loading the subsampled entire room as input without block/sphere subsampling.
    number of points per room in average, median, and std: (145841.0, 158783.87179487178, 84200.84445829492)
    """

    def __init__(
        self,
        data_root="data/ScanNet",
        split="train",
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        loop=1,
        presample=False,
        variable=False,
        n_shifted=1,
    ):
        super().__init__()
        self.split = split
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.presample = presample
        self.variable = variable
        self.loop = loop
        self.n_shifted = n_shifted
        self.pipe_transform = PointsToTensor()

        if split == "train" or split == "val":
            self.data_list = glob.glob(os.path.join(data_root, split, "*.pth"))
        elif split == "trainval":
            self.data_list = glob.glob(os.path.join(data_root, "train", "*.pth")) + glob.glob(
                os.path.join(data_root, "val", "*.pth")
            )
        elif split == "test":
            self.data_list = glob.glob(os.path.join(data_root, split, "*.pth"))
        else:
            raise ValueError("no such split: {}".format(split))

        logging.info("Totally {} samples in {} set.".format(len(self.data_list), split))

        processed_root = os.path.join(data_root, "processed")
        filename = os.path.join(processed_root, f"scannet_{split}_{voxel_size:.3f}.pkl")
        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data = []
            for item in tqdm(self.data_list, desc=f"Loading ScanNet {split} split"):
                data = torch.load(item)
                coord, feat, label = data[0:3]
                coord, feat, label = crop_pc(
                    coord,
                    feat,
                    label,
                    self.split,
                    self.voxel_size,
                    self.voxel_max,
                    variable=self.variable,
                )
                cdata = np.hstack((coord, feat, np.expand_dims(label, -1))).astype(np.float32)
                self.data.append(cdata)
            npoints = np.array([len(data) for data in self.data])
            logging.info(
                "split: %s, median npoints %.1f, avg num points %.1f, std %.1f"
                % (self.split, np.median(npoints), np.average(npoints), np.std(npoints))
            )
            os.makedirs(processed_root, exist_ok=True)
            with open(filename, "wb") as f:
                pickle.dump(self.data, f)
                print(f"{filename} saved successfully")
        elif presample:
            with open(filename, "rb") as f:
                self.data = pickle.load(f)
                print(f"{filename} load successfully")
            # median, average, std of number of points after voxel sampling for val set.
            # (100338.5, 109686.1282051282, 57024.51083415437)
            # before voxel sampling
            # (145841.0, 158783.87179487178, 84200.84445829492)

    def __getitem__(self, idx):
        data_idx = idx % len(self.data_list)
        if self.presample:
            coord, feat, label = np.split(self.data[data_idx], [3, 6], axis=1)
        else:
            data_path = self.data_list[data_idx]
            data = torch.load(data_path)
            coord, feat, label = data[0:3]

        feat = (feat + 1) * 127.5
        label = label.astype(np.long).squeeze()
        data = {
            "pos": coord.astype(np.float32),
            "x": feat.astype(np.float32),
            "y": label,
        }
        """debug 
        from openpoints.dataset import vis_multi_points
        import copy
        old_data = copy.deepcopy(data)
        if self.transform is not None:
            data = self.transform(data)
        data['pos'], data['x'], data['y'] = crop_pc(
            data['pos'], data['x'], data['y'], self.split, self.voxel_size, self.voxel_max,
            downsample=not self.presample, variable=self.variable)
            
        vis_multi_points([old_data['pos'][:, :3], data['pos'][:, :3]], colors=[old_data['x'][:, :3]/255.,data['x'][:, :3]])
        """
        if self.transform is not None:
            data = self.transform(data)

        if not self.presample:
            data["pos"], data["x"], data["y"] = crop_pc(
                data["pos"],
                data["x"],
                data["y"],
                self.split,
                self.voxel_size,
                self.voxel_max,
                downsample=not self.presample,
                variable=self.variable,
            )

        data = self.pipe_transform(data)

        if "heights" not in data.keys():
            data["heights"] = (
                data["pos"][:, self.gravity_dim : self.gravity_dim + 1]
                - data["pos"][:, self.gravity_dim : self.gravity_dim + 1].min()
            )
        return data

    def __len__(self):
        return len(self.data_list) * self.loop
