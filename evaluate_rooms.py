import argparse
import os
from typing import Dict, Optional

import fpsample
import numpy as np
import open3d as o3d
import pandas as pd
import torch
from loguru import logger
from numpy.typing import ArrayLike
from omegaconf import DictConfig
from tqdm import tqdm

from metrics.metrics import cd_unit_sphere, point_face_dist

MULTIPLIER = 10**3


@torch.no_grad()
def get_mectrics(args: DictConfig, gt: ArrayLike, pred: ArrayLike, gt_mesh: Optional[ArrayLike] = None) -> DictConfig:
    """
    Calculate the metrics for the given point clouds.

    Args:
        args (DictConfig): Configuration.
        gt (ArrayLike): Ground truth point cloud.
        pred (ArrayLike): Predicted point cloud.
        gt_mesh (ArrayLike, optional): Ground truth mesh. Defaults to None.

    Returns:
        DictConfig: Metrics.

    Raises:
        AssertionError: Ground truth mesh is required for SNPP dataset.
    """

    gt = torch.tensor(gt).float().cuda()

    data = {}

    # calculate metrics
    if args.dataset == "snpp":
        assert gt_mesh is not None, "Ground truth mesh is required for SNPP dataset"
        gt_faces = torch.tensor(np.array(gt_mesh.triangles)).long().cuda() if gt_mesh is not None else None
        gt_verts = torch.tensor(np.array(gt_mesh.vertices)).float().cuda() if gt_mesh is not None else None

        point_dist, face_dist = point_face_dist(pred, gt_verts, gt_faces, normalize=args.normalize)
        data["point_dist"] = point_dist * MULTIPLIER
        data["face_dist"] = face_dist * MULTIPLIER
    else:
        # We don't have ground truth mesh in ARKit
        data["point_dist"] = None
        data["face_dist"] = None

    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)

    cd_pred_gt, cd_gt_pred = cd_unit_sphere(pred, gt, normalize=args.normalize)

    data["cd_pred_gt"] = cd_pred_gt * MULTIPLIER
    data["cd_gt_pred"] = cd_gt_pred * MULTIPLIER

    return data


def calculate_model_metrics(data: Dict, model_name: str, args: DictConfig) -> Dict:
    """
    Calculate the metrics for the given model.

    Args:
        data (Dict): Data.
        model_name (str): Model name.
        args (DictConfig): Configuration.

    Returns:
        Dict: Model metrics.
    """
    faro = data["faro"]
    faro_mesh = data["faro_mesh"]
    segments = data["segments"]

    model_data = data["models"][model_name]
    model_metrics = {}

    logger.info("Calculating metrics for %s" % model_name)

    pbar = tqdm(model_data.keys())
    for model_config in pbar:
        pbar.set_description_str("Calculating metrics for %s" % model_config)
        pred = model_data[model_config]
        model_metrics[model_config] = {}
        metrics_data = get_mectrics(args, faro, pred, gt_mesh=faro_mesh, segments=segments)
        logger.info(metrics_data)
        model_metrics[model_config] = metrics_data

    return model_metrics


def load_folder_snpp(root: str, args: DictConfig) -> Optional[Dict]:
    """
    Load the data from the given folder our format.

    Args:
        root (str): Root folder.
        args (DictConfig): Configuration.

    Returns:
        Optional[Dict]: Data.
    """
    scans = os.path.join(root, "scans")
    iphone = os.path.join(scans, f"iphone{args.suffix}.ply")
    faro = os.path.join(scans, "mesh_aligned_0.05.ply")

    predictions = os.path.join(root, f"predictions{args.suffix}")
    if not os.path.exists(predictions):
        logger.warning("No predictions found in %s" % root)
        return None

    models = os.listdir(predictions)
    models = [m for m in models if m not in ["iphone", "gt", "tsdf"]]
    models = [os.path.join(predictions, model) for model in models]

    data = {"iphone": None, "faro": None, "faro_mesh": None, "models": {}}

    iphone_pcd = o3d.io.read_point_cloud(iphone)
    iphone_pcd = np.array(iphone_pcd.points)

    for model in models:
        logger.info("Loading data for %s" % model)
        model_predictions = [f for f in os.listdir(model) if f.endswith(".ply") or f.endswith(".xyz")]
        model_predictions = [os.path.join(model, f) for f in model_predictions]

        data["models"][model] = {}

        csv_name = f"metrics{args.suffix}.csv"
        model_metrics_path = os.path.join(model, csv_name)

        if os.path.exists(model_metrics_path):
            model_metrics = pd.read_csv(model_metrics_path)
            calculated_models = [v for v in model_metrics["model_config"].values if not pd.isna(v)]
        else:
            calculated_models = []

        for pred in model_predictions:
            name = pred.split("/")[-1][:-4]
            if name in calculated_models:
                logger.info(f"Metrics for {model}{name} already calculated")
                continue
            pred_pcd = np.array(o3d.io.read_point_cloud(pred).points)
            if iphone_pcd.shape[0] < pred_pcd.shape[0]:
                logger.warning(f"Downsampling {model} {name} due to different number of points")
                idxs = fpsample.bucket_fps_kdline_sampling(pred_pcd, iphone_pcd.shape[0], h=5)
                pred_pcd = pred_pcd[idxs]
            elif iphone_pcd.shape[0] > pred_pcd.shape[0]:
                logger.warning(f"Skipping {model} {name} due to different number of points")
                continue
            data["models"][model][name] = pred_pcd

    faro_mesh = o3d.io.read_triangle_mesh(faro)
    faro_pcd = o3d.io.read_point_cloud(faro)
    faro_pcd = np.array(faro_pcd.points)

    data["iphone"] = iphone_pcd
    data["faro"] = faro_pcd
    data["faro_mesh"] = faro_mesh

    logger.success("Loaded data from %s" % root)
    return data


def load_folder_arkit(root: str, args: DictConfig) -> Optional[Dict]:
    """
    Load the data from the given folder using our inference format.

    Args:
        root (str): Root folder.
        args (DictConfig): Configuration.

    Returns:
        Optional[Dict]: Data.
    """
    scans = os.path.join(root, "scans")
    faro = os.path.join(scans, "faro.ply")

    iphone = os.path.join(scans, f"iphone{args.suffix}.ply")
    predictions = os.path.join(root, f"predictions{args.suffix}")

    if not os.path.exists(predictions):
        logger.warning("No predictions found in %s" % root)
        return None

    models = os.listdir(predictions)
    models = [m for m in models if m not in ["iphone", "gt", "tsdf"]]
    models = [os.path.join(predictions, model) for model in models]

    data = {"iphone": None, "faro": None, "faro_mesh": None, "models": {}}

    for model in models:
        logger.info("Loading data for %s" % model)
        model_predictions = [f for f in os.listdir(model) if f.endswith(".ply") or f.endswith(".xyz")]
        model_predictions = [os.path.join(model, f) for f in model_predictions]

        data["models"][model] = {}

        csv_name = f"metrics{args.suffix}.csv"
        model_metrics_path = os.path.join(model, csv_name)

        if os.path.exists(model_metrics_path):
            model_metrics = pd.read_csv(model_metrics_path)
            calculated_models = model_metrics["model_config"].values
        else:
            calculated_models = []

        for pred in model_predictions:
            name = pred.split("/")[-1][:-4]
            if name in calculated_models:
                logger.info(f"Metrics for {model}{name} already calculated")
                continue
            data["models"][model][name] = np.array(o3d.io.read_point_cloud(pred).points)

    iphone_pcd = o3d.io.read_point_cloud(iphone)
    faro_mesh = o3d.io.read_triangle_mesh(faro)
    faro_pcd = o3d.io.read_point_cloud(faro)

    iphone_pcd = np.array(iphone_pcd.points)
    faro_pcd = np.array(faro_pcd.points)

    data["iphone"] = iphone_pcd
    data["faro"] = faro_pcd
    data["faro_mesh"] = faro_mesh

    logger.success("Loaded data from %s" % root)
    return data


def handle_scene(scene_folder: str, args: DictConfig) -> None:
    """
    Handle the scene folder. Calculate the metrics for the models in the scene.

    Args:
        scene_folder (str): Scene folder.
        args (DictConfig): Configuration.
    """
    if args.dataset == "snpp":
        data = load_folder_snpp(scene_folder, args)
    elif args.dataset == "arkit":
        data = load_folder_arkit(scene_folder, args)

    if data is None:
        return

    models = data["models"].keys()

    for model in models:
        csv_name = f"metrics{args.suffix}.csv"

        if args.normalize:
            csv_name += "_normalized.csv"

        metrics_path = os.path.join(scene_folder, "metrics", model, csv_name)

        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        model_metrics = calculate_model_metrics(data, model, args)

        # check if metrics were already calculated
        if os.path.exists(metrics_path):
            metrics = pd.read_csv(metrics_path)
            for model_config in model_metrics:
                if model_config in metrics["model_config"].values:
                    continue

                model_metrics_dict = model_metrics[model_config]
                model_metrics_dict["model_config"] = model_config
                metrics = pd.concat([metrics, pd.DataFrame([model_metrics_dict])], ignore_index=True)
        else:
            metrics = pd.DataFrame(columns=["model_config", "point_dist", "face_dist", "cd_pred_gt", "cd_gt_pred"])
            for model_config in model_metrics:
                save_cfg = model_metrics[model_config]
                save_cfg["model_config"] = model_config
                metrics = pd.concat([metrics, pd.DataFrame([model_metrics[model_config]])], ignore_index=True)

        # save the metrics
        metrics.to_csv(metrics_path, index=False)
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to the data directory.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the clustering.")
    parser.add_argument("--dataset", type=str, required=True, choices=["snpp", "arkit"], help="Dataset type.")
    parser.add_argument("--single_dir", action="store_true", help="Only single directional Metrics")
    parser.add_argument("--normalize", action="store_true", help="Normalize the point clouds")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    scene_folders = os.listdir(args.data_root)
    scene_folders = [os.path.join(args.data_root, f) for f in scene_folders]

    for scene_folder in scene_folders:
        handle_scene(scene_folder, args)


if __name__ == "__main__":
    main()
