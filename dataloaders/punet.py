import math
import numbers
import os
import random

import numpy as np
import pytorch3d.ops
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from tqdm.auto import tqdm

"""Most functions and classes are adapted from https://github.com/luost26/score-denoise"""


class NormalizeUnitSphere(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def normalize(pcl, center=None, scale=None):
        """
        Args:
            pcl:  The point cloud to be normalized, (N, 3)
        """
        if center is None:
            p_max = pcl.max(dim=0, keepdim=True)[0]
            p_min = pcl.min(dim=0, keepdim=True)[0]
            center = (p_max + p_min) / 2  # (1, 3)
        pcl = pcl - center
        if scale is None:
            scale = (pcl**2).sum(dim=1, keepdim=True).sqrt().max(dim=0, keepdim=True)[0]  # (1, 1)
        pcl = pcl / scale
        return pcl, center, scale

    def __call__(self, data):
        assert "pcl_noisy" not in data, "Point clouds must be normalized before applying noise perturbation."
        data["pcl_clean"], center, scale = self.normalize(data["pcl_clean"])
        data["center"] = center
        data["scale"] = scale
        return data


class AddNoise(object):
    def __init__(self, noise_std_min, noise_std_max):
        super().__init__()
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max

    def __call__(self, data):
        noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
        data["pcl_noisy"] = data["pcl_clean"] + torch.randn_like(data["pcl_clean"]) * noise_std
        data["noise_std"] = noise_std
        return data


class AddLaplacianNoise(object):
    def __init__(self, noise_std_min, noise_std_max):
        super().__init__()
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max

    def __call__(self, data):
        noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
        noise = torch.FloatTensor(np.random.laplace(0, noise_std, size=data["pcl_clean"].shape)).to(data["pcl_clean"])
        data["pcl_noisy"] = data["pcl_clean"] + noise
        data["noise_std"] = noise_std
        return data


class AddUniformBallNoise(object):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def __call__(self, data):
        N = data["pcl_clean"].shape[0]
        phi = np.random.uniform(0, 2 * np.pi, size=N)
        costheta = np.random.uniform(-1, 1, size=N)
        u = np.random.uniform(0, 1, size=N)
        theta = np.arccos(costheta)
        r = self.scale * u ** (1 / 3)

        noise = np.zeros([N, 3])
        noise[:, 0] = r * np.sin(theta) * np.cos(phi)
        noise[:, 1] = r * np.sin(theta) * np.sin(phi)
        noise[:, 2] = r * np.cos(theta)
        noise = torch.FloatTensor(noise).to(data["pcl_clean"])
        data["pcl_noisy"] = data["pcl_clean"] + noise
        return data


class AddCovNoise(object):
    def __init__(self, cov, std_factor=1.0):
        super().__init__()
        self.cov = torch.FloatTensor(cov)
        self.std_factor = std_factor

    def __call__(self, data):
        num_points = data["pcl_clean"].shape[0]
        noise = np.random.multivariate_normal(np.zeros(3), self.cov.numpy(), num_points)  # (N, 3)
        noise = torch.FloatTensor(noise).to(data["pcl_clean"])
        data["pcl_noisy"] = data["pcl_clean"] + noise * self.std_factor
        data["noise_std"] = self.std_factor
        return data


class AddDiscreteNoise(object):
    def __init__(self, scale, prob=0.1):
        super().__init__()
        self.scale = scale
        self.prob = prob
        self.template = np.array(
            [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ],
            dtype=np.float32,
        )

    def __call__(self, data):
        num_points = data["pcl_clean"].shape[0]
        uni_rand = np.random.uniform(size=num_points)
        noise = np.zeros([num_points, 3])
        for i in range(self.template.shape[0]):
            idx = np.logical_and(0.1 * i <= uni_rand, uni_rand < 0.1 * (i + 1))
            noise[idx] = self.template[i].reshape(1, 3)
        noise = torch.FloatTensor(noise).to(data["pcl_clean"])
        # print(data['pcl_clean'])
        # print(self.scale)
        data["pcl_noisy"] = data["pcl_clean"] + noise * self.scale
        data["noise_std"] = self.scale
        return data


class RandomScale(object):
    def __init__(self, scales):
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales

    def __call__(self, data):
        scale = random.uniform(*self.scales)
        data["pcl_clean"] = data["pcl_clean"] * scale
        if "pcl_noisy" in data:
            data["pcl_noisy"] = data["pcl_noisy"] * scale
        return data


class RandomRotate(object):
    def __init__(self, degrees=180.0, axis=0):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis

    def __call__(self, data):
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if self.axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif self.axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        matrix = torch.tensor(matrix)

        data["pcl_clean"] = torch.matmul(data["pcl_clean"], matrix)
        if "pcl_noisy" in data:
            data["pcl_noisy"] = torch.matmul(data["pcl_noisy"], matrix)

        return data


def standard_train_transforms(noise_std_min, noise_std_max, scale_d=0.2, rotate=True):
    transforms = [
        NormalizeUnitSphere(),
        AddNoise(noise_std_min=noise_std_min, noise_std_max=noise_std_max),
        RandomScale([1.0 - scale_d, 1.0 + scale_d]),
    ]
    if rotate:
        transforms += [
            RandomRotate(axis=0),
            RandomRotate(axis=1),
            RandomRotate(axis=2),
        ]
    return Compose(transforms)


def standard_train_transforms_clean(scale_d=0.2, rotate=True):
    transforms = [
        NormalizeUnitSphere(),
        RandomScale([1.0 - scale_d, 1.0 + scale_d]),
    ]
    if rotate:
        transforms += [
            RandomRotate(axis=0),
            RandomRotate(axis=1),
            RandomRotate(axis=2),
        ]
    return Compose(transforms)


# taken from [ScoreDenoise]
class PointCloudDataset(Dataset):
    def __init__(self, root, dataset, split, resolution, transform=None):
        super().__init__()
        self.pcl_dir = os.path.join(root, dataset, "pointclouds", split, resolution)
        self.transform = transform
        self.pointclouds = []
        self.pointcloud_names = []
        for fn in tqdm(os.listdir(self.pcl_dir), desc="Loading"):
            if fn[-3:] != "xyz":
                continue
            pcl_path = os.path.join(self.pcl_dir, fn)
            if not os.path.exists(pcl_path):
                raise FileNotFoundError("File not found: %s" % pcl_path)
            pcl = torch.FloatTensor(np.loadtxt(pcl_path, dtype=np.float32))
            self.pointclouds.append(pcl)
            self.pointcloud_names.append(fn[:-4])

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {"pcl_clean": self.pointclouds[idx].clone(), "name": self.pointcloud_names[idx]}
        if self.transform is not None:
            data = self.transform(data)
        return data


def standard_train_transforms(noise_std_min, noise_std_max, scale_d=0.2, rotate=True):
    transforms = [
        NormalizeUnitSphere(),
        AddNoise(noise_std_min=noise_std_min, noise_std_max=noise_std_max),
        RandomScale([1.0 - scale_d, 1.0 + scale_d]),
    ]
    if rotate:
        transforms += [
            RandomRotate(axis=0),
            RandomRotate(axis=1),
            RandomRotate(axis=2),
        ]
    return Compose(transforms)


def standard_train_transforms_clean(scale_d=0.2, rotate=True):
    transforms = [
        NormalizeUnitSphere(),
        RandomScale([1.0 - scale_d, 1.0 + scale_d]),
    ]
    if rotate:
        transforms += [
            RandomRotate(axis=0),
            RandomRotate(axis=1),
            RandomRotate(axis=2),
        ]
    return Compose(transforms)


def get_dataset(
    dataset_root,
    split,
    dataset="PUNet",
    noise_min=0.010,
    noise_max=0.020,
    aug_rotate=True,
    patch_size=2048,
    resolutions=["10000_poisson", "30000_poisson", "50000_poisson"],
):
    if noise_max > 0:
        transform = standard_train_transforms(noise_std_max=noise_max, noise_std_min=noise_min, rotate=aug_rotate)
    else:
        transform = standard_train_transforms_clean(rotate=aug_rotate)

    ds = PairedPatchDataset(
        datasets=[
            PointCloudDataset(root=dataset_root, dataset=dataset, split=split, resolution=resl, transform=transform)
            for resl in resolutions
        ],
        patch_size=patch_size,
        patch_ratio=1.0,
        on_the_fly=True,
    )
    return ds


def create_collate_fn(aligner):
    def align_patches(data):
        aligned_data = {}
        for d in data:
            clean = d["hr_points"]
            noisy = d["lr_points"]
            dis, alignment = aligner(noisy.unsqueeze(0), clean.unsqueeze(0), 0.01, 100)
            clean = clean[alignment]
            aligned_data["lr_points"] = noisy
            aligned_data["hr_points"] = clean
            aligned_data["center"] = d["center"]
            aligned_data["scale"] = d["scale"]
        return aligned_data

    return align_patches


def get_alignment_clean(aligner):
    @torch.no_grad()
    def align(noisy, clean):
        noisy = noisy.clone().transpose(1, 2).contiguous()
        clean = clean.clone().transpose(1, 2).contiguous()
        dis, alignment = aligner(noisy, clean, 0.01, 100)
        return alignment.detach()

    return align


def make_patches_for_pcl_pair(pcl_A, pcl_B, patch_size, num_patches, ratio):
    """
    Args:
        pcl_A:  The first point cloud, (N, 3).
        pcl_B:  The second point cloud, (rN, 3).
        patch_size:   Patch size M.
        num_patches:  Number of patches P.
        ratio:    Ratio r.
    Returns:
        (P, M, 3), (P, rM, 3)
    """
    N = pcl_A.size(0)
    seed_idx = torch.randperm(N)[:num_patches]  # (P, )
    seed_pnts = pcl_A[seed_idx].unsqueeze(0)  # (1, P, 3)
    _, _, pat_A = pytorch3d.ops.knn_points(
        seed_pnts, pcl_A.unsqueeze(0), K=patch_size, return_nn=True, return_sorted=False
    )
    pat_A = pat_A[0]  # (P, M, 3)
    _, _, pat_B = pytorch3d.ops.knn_points(
        seed_pnts, pcl_B.unsqueeze(0), K=int(ratio * patch_size), return_nn=True, return_sorted=False
    )
    pat_B = pat_B[0]
    return pat_A, pat_B


class PairedPatchDataset(Dataset):
    def __init__(self, datasets, patch_ratio, on_the_fly=True, patch_size=1000, num_patches=1000, transform=None):
        super().__init__()
        self.datasets = datasets
        self.len_datasets = sum([len(dset) for dset in datasets])
        self.patch_ratio = patch_ratio
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.on_the_fly = on_the_fly
        self.transform = transform
        self.patches = []
        # Initialize
        if not on_the_fly:
            self.make_patches()

    def make_patches(self):
        for dataset in tqdm(self.datasets, desc="MakePatch"):
            for data in tqdm(dataset):
                pat_noisy, pat_clean = make_patches_for_pcl_pair(
                    data["pcl_noisy"],
                    data["pcl_clean"],
                    patch_size=self.patch_size,
                    num_patches=self.num_patches,
                    ratio=self.patch_ratio,
                )  # (P, M, 3), (P, rM, 3)
                for i in range(pat_noisy.size(0)):
                    self.patches.append(
                        (
                            pat_noisy[i],
                            pat_clean[i],
                        )
                    )

    def __len__(self):
        if not self.on_the_fly:
            return len(self.patches)
        else:
            return self.len_datasets * self.num_patches

    def __getitem__(self, idx):
        if self.on_the_fly:
            pcl_dset = random.choice(self.datasets)
            pcl_data = pcl_dset[idx % len(pcl_dset)]
            pat_noisy, pat_clean = make_patches_for_pcl_pair(
                pcl_data["pcl_noisy"],
                pcl_data["pcl_clean"],
                patch_size=self.patch_size,
                num_patches=1,
                ratio=self.patch_ratio,
            )
            data = {"pcl_noisy": pat_noisy[0], "pcl_clean": pat_clean[0]}
        else:
            data = {
                "pcl_noisy": self.patches[idx][0].clone(),
                "pcl_clean": self.patches[idx][1].clone(),
            }

        if self.transform is not None:
            data = self.transform(data)

        # centering
        center = data["pcl_clean"].mean(dim=0)
        data["pcl_noisy"] -= center
        data["pcl_clean"] -= center

        # scale to unit sphere
        scale = torch.max(torch.norm(data["pcl_noisy"], dim=1))
        data["pcl_noisy"] /= scale
        data["pcl_clean"] /= scale

        new_data = {
            "noisy_points": data["pcl_noisy"],
            "clean_points": data["pcl_clean"],
            "center": center,
            "scale": scale,
        }
        return new_data


class PairedPatchDatasetNPZ(Dataset):
    def __init__(self, root, split="train") -> None:
        super().__init__()
        self.files = os.listdir(os.path.join(root, split))
        self.split = split
        self.root = root

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = np.load(os.path.join(self.root, self.split, file))
        data = {
            "noisy_points": torch.FloatTensor(data["lr_points"]),
            "clean_points": torch.FloatTensor(data["hr_points"]),
            "center": torch.FloatTensor(data["center"]),
            "scale": torch.FloatTensor(data["scale"]),
        }
        return data
