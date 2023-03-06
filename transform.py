import os
import utils
import argparse

import numpy as np
import open3d as o3d

ply_vertex_type = np.dtype(
    [
        ("x", "f4"), 
        ("y", "f4"),
        ("z", "f4"), 
        ("red", "u1"), 
        ("green", "u1"), 
        ("blue", "u1"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4")
    ]
)

DATA_ROOT = "/home/hm/fuguiduo/datasets/3DMatch-FCGF/npz"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd0", type=str, help="name of point cloud 0")
    parser.add_argument("--pcd1", type=str, help="name of point cloud 1")
    args = parser.parse_args()

    points0 = utils.npz2npy(os.path.join(DATA_ROOT, args.pcd0))
    points1 = utils.npz2npy(os.path.join(DATA_ROOT, args.pcd1))
    points0[:,3:6] = np.ones((len(points0), 3), dtype=np.int32)[:] * np.array([255, 255, 0], dtype=np.int32)
    points1[:,3:6] = np.ones((len(points1), 3), dtype=np.int32)[:] * np.array([0, 255, 255], dtype=np.int32)

    points1, rotmat, transd = utils.transform_augment(points1, 30.0, 4.0)
    utils.log_info(f"augmented, angle: {utils.resolve_axis_angle(rotmat)}, dist: {transd}")

    points0 = utils.npy2o3d(points0)
    points1 = utils.npy2o3d(points1)
    points0.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.1, max_nn=30))
    points1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.1, max_nn=30))

    points0 = utils.o3d2npy(points0)
    points1 = utils.o3d2npy(points1)

    utils.dump1frag(points0, ply_vertex_type, out_name="pcd0.ply")
    utils.dump1frag(points1, ply_vertex_type, out_name="pcd1.ply")
    utils.fuse2frags(points0, points1, ply_vertex_type, out_name="fuse.ply")