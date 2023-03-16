import os
import utils
import argparse
from tqdm import tqdm

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threedmatch_root", type=str, default="/home/hm/fuguiduo/datasets/3DMatch-FCGF", help="dir that has txts and npzs")
    parser.add_argument("--out_root", type=str, default="samples/augment", help="dir to store results")
    args = parser.parse_args()

    data_root = args.threedmatch_root
    out_root = args.out_root
    
    txts = sorted(os.listdir(os.path.join(data_root, "txt")))
    txts = np.random.choice(txts, size=len(txts), replace=False)
    
    with open(os.path.join(data_root, "txt", txts[0]), 'r') as f:
        pair_names = f.readlines()
    
    for line in tqdm(pair_names, total=len(pair_names), ncols=100, desc=txts[0]):
        pair_name = line.rstrip().split(' ')
        points0 = utils.npz2npy(os.path.join(data_root, "npz", pair_name[0]))
        points1 = utils.npz2npy(os.path.join(data_root, "npz", pair_name[1]))

        points0[:,3:6] = np.ones((len(points0), 3), dtype=np.int32)[:] * np.array([255, 255, 0], dtype=np.int32)
        points1[:,3:6] = np.ones((len(points1), 3), dtype=np.int32)[:] * np.array([0, 255, 255], dtype=np.int32)

        points1, rotmat, transd = utils.transform_augment(points1, 30.0, 4.0)
        raxis, angle_rad = utils.resolve_axis_angle(rotmat)
        angle_deg = angle_rad / np.pi * 180.0
        transd = np.sqrt(np.sum(transd ** 2))
        tqdm.write(f"augmented angle degree: {angle_deg:5.2f}, dist: {transd:5.2f}")

        points0 = utils.npy2o3d(points0)
        points1 = utils.npy2o3d(points1)
        points0.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.1, max_nn=30))
        points1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.1, max_nn=30))

        points0 = utils.o3d2npy(points0)
        points1 = utils.o3d2npy(points1)

        ply0_name = pair_name[0].split('@')[1][:-4]
        ply1_name = pair_name[1].split('@')[1][:-4]
        out_dir = os.path.join(out_root, f"{ply0_name}{ply1_name}@{angle_deg:.2f}_{transd:.2f}")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, mode=0o775)
        utils.dump1frag(points0, ply_vertex_type, out_dir, out_name=f"{ply0_name}.ply")
        utils.dump1frag(points1, ply_vertex_type, out_dir, out_name=f"{ply1_name}.ply")
        utils.fuse2frags(points0, points1, ply_vertex_type, out_dir, out_name="fuse.ply")