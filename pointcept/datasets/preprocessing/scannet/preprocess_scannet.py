"""
Preprocessing Script for ScanNet 20/200

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import laspy
import os
import argparse
import glob
import json
import plyfile
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
# import open3d as o3d


# Load external constants
from meta_data.scannet200_constants import VALID_CLASS_IDS_200, VALID_CLASS_IDS_20

CLOUD_FILE_PFIX = "_vh_clean_2"
SEGMENTS_FILE_PFIX = ".0.010000.segs.json"
AGGREGATIONS_FILE_PFIX = ".aggregation.json"
CLASS_IDS200 = VALID_CLASS_IDS_200
CLASS_IDS20 = VALID_CLASS_IDS_20
IGNORE_INDEX = -1

def face_normal(vertex, face):
    v01 = vertex[face[:, 1]] - vertex[face[:, 0]]
    v02 = vertex[face[:, 2]] - vertex[face[:, 0]]
    vec = np.cross(v01, v02)
    length = np.sqrt(np.sum(vec**2, axis=1, keepdims=True)) + 1.0e-8
    nf = vec / length
    area = length * 0.5
    return nf, area

def vertex_normal(vertex, face):
    nf, area = face_normal(vertex, face)
    nf = nf * area

    nv = np.zeros_like(vertex)
    for i in range(face.shape[0]):
        nv[face[i]] += nf[i]

    length = np.sqrt(np.sum(nv**2, axis=1, keepdims=True)) + 1.0e-8
    nv = nv / length
    return nv


def handle_process(
    scene_path, output_path, train_scenes, val_scenes, parse_normals=True
):
    scene_id = os.path.basename(scene_path).split(".")[0]

    # print(f"Processing: {scene_id}")
    # print(f"Processing train scenes: {train_scenes}")
    

    if scene_id in train_scenes:
        # output_path = os.path.join(output_path, "train", f"{scene_id}")
        output_path = os.path.join(output_path, "train", f"{scene_id}")

        split_name = "train"
    elif scene_id in val_scenes:
        output_path = os.path.join(output_path, "val", f"{scene_id}")
        split_name = "val"
    else:
        output_path = os.path.join(output_path, "test", f"{scene_id}")
        split_name = "test"

    print(f"Processing: {scene_id} in {split_name}")

    # vertices, faces = read_plymesh(mesh_path)
    # load las
    # las_path = os.path.join(scene_path, f"{scene_id}{CLOUD_FILE_PFIX}.las")
    las_path = scene_path
    las = laspy.read(las_path)
    vertices = np.vstack((las.x, las.y, las.z)).T
    coords = vertices[:, :3]

    # pcd = o3d.geometry.PointCloud()

    # # Assign the points to the PointCloud object
    # pcd.points = o3d.utility.Vector3dVector(coords)

    # # Estimate the normals
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # # Normalize the normals (optional)
    # pcd.normalize_normals()

    # # Visualize the point cloud with normals
    # normals = o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    # normals = np.asarray(normals).astype(np.float32)


    colors = np.zeros_like(coords, dtype=np.uint8)
    save_dict = dict(
        coord=coords.astype(np.float32),
        color=colors.astype(np.uint8),
    )

    # Parse Normals
    if parse_normals:
        # replace normals with zeros
        save_dict["normal"] = np.zeros_like(coords).astype(np.float32)
        # replace normals with zeros


        
    # Load segments file
    if split_name != "test":

        # Generate new labels
        semantic_gt20 = np.ones((vertices.shape[0]), dtype=np.int16) * IGNORE_INDEX
        instance_ids = np.ones((vertices.shape[0]), dtype=np.int16) * IGNORE_INDEX
       
        save_dict["segment20"] = semantic_gt20

        save_dict["instance"] = instance_ids

        # Concatenate with original cloud
        processed_vertices = np.hstack((semantic_gt20, instance_ids))

        if np.any(np.isnan(processed_vertices)) or not np.all(
            np.isfinite(processed_vertices)
        ):
            raise ValueError(f"Find NaN in Scene: {scene_id}")

    # Save processed data
    os.makedirs(output_path, exist_ok=True)
    for key in save_dict.keys():
        np.save(os.path.join(output_path, f"{key}.npy"), save_dict[key])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ScanNet dataset containing scene folders",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val folders will be located",
    )
    parser.add_argument(
        "--parse_normals", default=True, type=bool, help="Whether parse point normals"
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    config = parser.parse_args()

    # Load label map
    labels_pd = pd.read_csv(
        "pointcept/datasets/preprocessing/scannet/meta_data/scannetv2-labels.combined.tsv",
        sep="\t",
        header=0,
    )

    # Load train/val splits
    with open(
        "pointcept/datasets/preprocessing/scannet/meta_data/forest_train.txt"
    ) as train_file:
        train_scenes = train_file.read().splitlines()
    with open(
        "pointcept/datasets/preprocessing/scannet/meta_data/forest_val.txt"
    ) as val_file:
        val_scenes = val_file.read().splitlines()

    # Create output directories
    train_output_dir = os.path.join(config.output_root, "train")
    os.makedirs(train_output_dir, exist_ok=True)
    val_output_dir = os.path.join(config.output_root, "val")
    os.makedirs(val_output_dir, exist_ok=True)
    test_output_dir = os.path.join(config.output_root, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    # Load scene paths
    scene_paths = sorted(glob.glob(config.dataset_root + "/*"))

    # Preprocess data.
    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    _ = list(
        pool.map(
            handle_process,
            scene_paths,
            repeat(config.output_root),
            repeat(train_scenes),
            repeat(val_scenes),
            repeat(config.parse_normals),
        )
    )
