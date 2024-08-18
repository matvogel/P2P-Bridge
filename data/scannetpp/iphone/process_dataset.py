# This script combines code from the original scannetpp repo to process and the iphone scans.
import argparse
import gc
import json
import os
import shutil
import zlib

import imageio as iio
import lz4.block
import numpy as np
import open3d.core as o3c
from common.scene_release import ScannetppScene_Release
from common.utils.utils import *
from iphone.arkit_pcl import *
from PIL import Image
from tqdm import tqdm


def extract_rgb(scene, sr=None, downscale=True):
    if sr is None:
        scene.iphone_rgb_dir.mkdir(parents=True, exist_ok=True)
        cmd = f"ffmpeg -hide_banner -i {scene.iphone_video_path} -threads 8 -start_number 0 {scene.iphone_rgb_dir}/frame_%06d.png"
        run_command(cmd, verbose=True)
    else:
        scene.iphone_rgb_dir.mkdir(parents=True, exist_ok=True)
        cmd = f'ffmpeg -hide_banner -i {scene.iphone_video_path} -threads 8 -start_number 0 -vf "select=not(mod(n\,{sr})),scale=256:192" -vsync vfr {scene.iphone_rgb_dir}/frame_%06d.png'
        run_command(cmd, verbose=True)

        files = [f for f in os.listdir(scene.iphone_rgb_dir) if f.endswith(".png")]

        # rename the frames such that they mach the actual frame numbering (we reduced it by a factor of sr)
        for f in files:
            f_path = scene.iphone_rgb_dir / f
            file_idx = int(f.split("_")[-1].split(".")[0])
            idx_true = file_idx * sr
            new_filename = scene.iphone_rgb_dir / f"tmp_frame_{idx_true:06}.png"
            shutil.move(f_path, new_filename)

        # remove tmp in name
        files = [f for f in os.listdir(scene.iphone_rgb_dir) if f.endswith(".png")]
        for f in files:
            f_path = scene.iphone_rgb_dir / f
            new_filename = scene.iphone_rgb_dir / f.replace("tmp_", "")
            shutil.move(f_path, new_filename)


def extract_masks(scene):
    scene.iphone_video_mask_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -hide_banner -i {str(scene.iphone_video_mask_path)} -pix_fmt gray -start_number 0 {scene.iphone_video_mask_dir}/frame_%06d.png"
    run_command(cmd, verbose=True)


def extract_depth(scene, sample_rate=1):
    # global compression with zlib
    height, width = 192, 256
    scene.iphone_depth_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(scene.iphone_depth_path, "rb") as infile:
            data = infile.read()
            data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
            depth = np.frombuffer(data, dtype=np.float32).reshape(-1, height, width)

        for frame_id in tqdm(range(0, depth.shape[0], sample_rate), desc="decode_depth"):
            iio.imwrite(
                f"{scene.iphone_depth_dir}/frame_{frame_id:06}.png",
                (depth * 1000).astype(np.uint16),
            )

    # per frame compression with lz4/zlib
    except Exception:
        frame_id = 0
        with open(scene.iphone_depth_path, "rb") as infile:
            while True:
                size = infile.read(4)  # 32-bit integer
                if len(size) == 0:
                    break
                size = int.from_bytes(size, byteorder="little")
                if frame_id % sample_rate != 0:
                    infile.seek(size, 1)
                    frame_id += 1
                    continue

                # read the whole file
                data = infile.read(size)
                try:
                    # try using lz4
                    data = lz4.block.decompress(data, uncompressed_size=height * width * 2)  # UInt16 = 2bytes
                    depth = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
                except:
                    # try using zlib
                    data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
                    depth = np.frombuffer(data, dtype=np.float32).reshape(height, width)
                    depth = (depth * 1000).astype(np.uint16)

                # 6 digit frame id = 277 minute video at 60 fps
                iio.imwrite(f"{scene.iphone_depth_dir}/frame_{frame_id:06}.png", depth)
                frame_id += 1


def process_frame(frame_id, data, iphone_depth_dir, iphone_rgb_dir, args, rgb=None, depth=None):
    camera_to_world = np.array(data["aligned_pose"]).reshape(4, 4)
    intrinsic = np.array(data["intrinsic"]).reshape(3, 3)
    rgb = (
        np.array(Image.open(os.path.join(iphone_rgb_dir, frame_id + ".png")), dtype=np.uint8)
        if rgb is None
        else rgb[frame_id]
    )
    depth = (
        np.array(
            Image.open(os.path.join(iphone_depth_dir, frame_id + ".png")),
            dtype=np.float32,
        )
        / 1000.0
        if depth is None
        else depth[frame_id]
    )

    pcd = backproject(
        rgb,
        depth,
        camera_to_world,
        intrinsic,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        use_point_subsample=False,
        use_voxel_subsample=True,
        voxel_grid_size=args.grid_size,
        outlier_radius=args.outlier_radius,
        n_outliers=args.n_outliers,
        no_filtering=args.no_cleaning,
    )
    return pcd.to_legacy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--filename", type=str, default="iphone", help="Name of the iphone scans.")
    parser.add_argument("--split", type=int, default=None, help="Split id to process")
    parser.add_argument("--sample_rate", type=int, default=30, help="Sample rate of the frames.")
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument(
        "--grid_size",
        type=float,
        default=0.01,
        help="Grid size for voxel downsampling.",
    )
    parser.add_argument(
        "--n_outliers",
        type=int,
        default=10,
        help="Number of neighbors for outlier removal.",
    )
    parser.add_argument("--outlier_radius", type=float, default=0.05, help="Radius for outlier removal.")
    parser.add_argument(
        "--final_grid_size",
        type=float,
        default=0.01,
        help="Grid size for voxel downsampling.",
    )
    parser.add_argument(
        "--final_n_outliers",
        type=int,
        default=10,
        help="Number of neighbors for outlier removal.",
    )
    parser.add_argument(
        "--final_outlier_radius",
        type=float,
        default=0.05,
        help="Radius for outlier removal.",
    )
    parser.add_argument(
        "--no_cleaning",
        action="store_true",
        help="Do not apply any outlier removal etc.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    scenes_root = args.data_root
    scene_ids = [item for item in os.listdir(scenes_root) if os.path.isdir(os.path.join(scenes_root, item))]

    scenes_filtered = []

    # check if we have a faro scan inside
    for scene_id in scene_ids:
        scene_path = os.path.join(scenes_root, scene_id)
        faro_path = os.path.join(scene_path, "scans", "mesh_aligned_0.05.ply")
        if os.path.exists(faro_path):
            scenes_filtered.append(scene_id)

    scenes_filtered.sort()

    # create 10 splits
    batch_size = int(np.ceil(len(scenes_filtered) / 10))
    if args.split is not None:
        scenes_filtered = scenes_filtered[args.split * batch_size : (args.split + 1) * batch_size]

    # process the scenes
    for scene_idx, scene_id in tqdm(enumerate(scenes_filtered), desc="Scenes", total=len(scenes_filtered)):
        # extract the frames and depth
        print("#" * 50)
        print("Processing scene: ", scene_id)
        scene = ScannetppScene_Release(scene_id, data_root=scenes_root)
        iphone_scan_path = os.path.join(scene.data_root, scene_id, "scans", f"{args.filename}.ply")

        if os.path.exists(iphone_scan_path) and not args.overwrite:
            print("Skipping", scene_id)
            continue

        print("Extracting frames and depth")
        extract_rgb(scene, sr=args.sample_rate)
        print("Extracted RGB")

        print("Extracting masks")
        extract_depth(scene, sample_rate=args.sample_rate)
        print("Extracted Depth")

        iphone_rgb_dir = scene.iphone_rgb_dir
        iphone_depth_dir = scene.iphone_depth_dir

        with open(scene.iphone_pose_intrinsic_imu_path, "r") as f:
            json_data = json.load(f)
        frame_data = [(frame_id, data) for frame_id, data in json_data.items()]
        frame_data.sort()

        frame_data = frame_data[:: args.sample_rate]

        all_xyz = []
        all_rgb = []

        for frame_id, data in tqdm(frame_data, desc="Processing frames"):
            pcd = process_frame(frame_id, data, iphone_depth_dir, iphone_rgb_dir, args)
            colors = np.array(pcd.colors)
            points = np.array(pcd.points)
            all_xyz.append(points)
            all_rgb.append(colors)
            o3c.cuda.release_cache()

        all_xyz = np.concatenate(all_xyz, axis=0)
        all_rgb = np.concatenate(all_rgb, axis=0)

        # move back to gpu
        full_pcd = create_gpu_pointcloud(xyz=all_xyz, rgb=all_rgb)

        print("Voxel downsampling. Number of points: ", full_pcd)
        full_pcd = full_pcd.voxel_down_sample(args.final_grid_size)

        # full_pcd = outlier_removal_gpu(full_pcd, nb_points=args.final_n_outliers, radius=args.final_outlier_radius)
        full_pcd = full_pcd.to_legacy()

        if not args.no_cleaning:
            print("Removing outliers. Number of points: ", full_pcd)
            all_xyz = np.array(full_pcd.points)
            all_rgb = np.array(full_pcd.colors)
            all_xyz, all_rgb, _ = outlier_removal_cuml(
                xyz=all_xyz,
                rgb=all_rgb,
                nb_points=args.final_n_outliers,
                radius=args.final_outlier_radius,
                std_thresh=2.0,
            )

            # load faro scan to do post filtering
            faro_scan_path = os.path.join(scene.data_root, scene_id, "scans", "mesh_aligned_0.05.ply")
            faro_scan = o3d.io.read_point_cloud(faro_scan_path)
            faro_xyz = np.array(faro_scan.points)

            all_xyz, all_rgb, *_ = filter_iphone_scan_fast(all_xyz, all_rgb, faro_xyz)

        save_point_cloud(
            filename=iphone_scan_path,
            points=all_xyz,
            rgb=all_rgb,
            binary=True,
            verbose=False,
        )

        print("Saved point cloud with {} points to {}".format(all_xyz.shape[0], iphone_scan_path))
        o3c.cuda.release_cache()

        # remove the extracted frames and depth
        # os.system("rm -rf {}".format(iphone_rgb_dir))
        # os.system("rm -rf {}".format(iphone_depth_dir))

        gc.collect()


if __name__ == "__main__":
    main()
