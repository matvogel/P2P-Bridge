import argparse
import gc
import os
import traceback
from typing import List

import numpy as np
import torch
from loguru import logger
from numpy.typing import ArrayLike
from omegaconf import DictConfig
from processing.image_features import load_dino, process_scene
from torch.multiprocessing import spawn


def mp_wrapper(id: int, scene_batches: ArrayLike, args: DictConfig) -> None:
    """
    Wrapper function for multiprocessing.

    Args:
        id (int): ID of the process.
        scene_batches (ArrayLike): List of scene IDs to process.
        args (DictConfig): Configuration dictionary.
    """
    scenes = scene_batches[id]
    handle_scenes(scenes, args)


def handle_scenes(scene_ids: List, args: DictConfig) -> None:
    """
    Handle a list of scenes.

    Args:
        scene_ids (List): List of scene IDs to process.
        args (DictConfig): Configuration dictionary.
    """
    feature_type = args.feature_type

    # create extractors
    if feature_type == "rgb":
        f_shape = 3
    elif feature_type == "dino":
        f_shape = 384
        model = load_dino(args.dino_model_name)
        model = torch.compile(model, mode="reduce-overhead")
    else:
        raise ValueError(f"Feature type not supported: {feature_type}! Use 'rgb' or 'dino'.")

    for scene_id in scene_ids:
        target_path = os.path.join(
            args.output_root,
            scene_id,
            "features",
            f"{args.feature_type}_{args.source_cloud}{args.feature_suffix}",
        )
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        try:
            logger.info("Processing scene", scene_id)
            process_scene(
                model=model,
                f_shape=f_shape,
                scene_id=scene_id,
                data_root=args.data_root,
                target_path=target_path,
                feature_type=args.feature_type,
                feature_suffix=args.feature_suffix,
                sampling_rate=args.sampling_rate,
                image_width=args.image_width,
                image_height=args.image_height,
                overwrite=args.overwrite,
                pointcloud_source=args.source_cloud,
                downscale=True,
                autoskip=True,
                batch_size=args.batch_size,
            )
            logger.info("Done with scene", scene_id)
            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            logger.info(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing features.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="dino",
        help="Type of features to extract.",
        choices=["rgb", "dino"],
    )
    parser.add_argument(
        "--feature_suffix",
        type=str,
        default="",
        help="Suffix to add to the feature name.",
    )
    parser.add_argument(
        "--source_cloud",
        type=str,
        default="iphone",
        help="Source of the pointcloud coordinates.",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=10,
        help="Number of scans to skip between each scan.",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=256,
        help="Width of the images in the video.",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=192,
        help="Height of the images in the video.",
    )
    parser.add_argument(
        "--dino_model_name",
        type=str,
        default="dinov2_vits14",
        help="Name of the DINO model to use.",
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=1,
        help="Number of processes to use.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for feature extraction.",
    )
    args = parser.parse_args()

    root_dir = os.listdir(args.data_root)
    scenes = [f for f in root_dir if os.path.isdir(os.path.join(args.data_root, f))]

    if args.output_root is None:
        args.output_root = args.data_root

    logger.info("Processing", len(scenes), "scenes")
    scene_batches = np.array_split(scenes, args.nprocs)
    spawn(mp_wrapper, args=(scene_batches, args), nprocs=args.nprocs, join=True)


if __name__ == "__main__":
    main()
