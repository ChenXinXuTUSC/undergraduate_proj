import os
import numpy as np
from plyfile import PlyData, PlyElement
import open3d as o3d
from tqdm import tqdm

import config
import utils

# step1: read point cloud pair
# step2: voxel down sample
# step3: extract ISS feature
# step4: FPFH feature description
# step5: RANSAC registration
#   step5.1: establish feature correspondences using KNN
#   step5.2: select 4 pairs to solve the transformation R and t
#   step5.3: repeat step5.2 until error converge
# step6: refine estimation for ICP

if __name__ == "__main__":
    args = vars(config.args)
    npzs = [os.path.join(args["3dmatch_root"], "npz", file) for file in sorted(os.listdir(os.path.join(args["3dmatch_root"], "npz")))]
    txts = [os.path.join(args["3dmatch_root"], "txt", file) for file in sorted(os.listdir(os.path.join(args["3dmatch_root"], "txt")))]

    # generate some demo pointcloud
    with open(txts[0], 'r') as f:
        lines = f.readlines()
        lines = [line.rstrip().split(' ') for line in lines]
        for line in lines:
            #
            #
            
            points1 = utils.npz2ply(os.path.join(args["3dmatch_root"], "npz", line[0]) , overwrite_rgb=True, new_rgb=[255, 0, 0])
            points2 = utils.npz2ply(os.path.join(args["3dmatch_root"], "npz", line[1]) , overwrite_rgb=True, new_rgb=[0, 255, 0])
            points1 = utils.voxel_down_sample(points1, 0.05)
            points2 = utils.voxel_down_sample(points2, 0.05)
            # points1, rotmat, transd = utils.transform_augment(points1, 60.0, 2.0)
            # print(f"{line[0]}:{points1.shape}, {line[1]}:{points2.shape}, overlap ratio:{line[2]}")

            keypoints1 = utils.iss_detect(points1)
            keypoints2 = utils.iss_detect(points2)
            points1[keypoints1["id"].values, 3:] = np.array([0, 255, 255])
            points2[keypoints2["id"].values, 3:] = np.array([0, 255, 255])

            out_dir  = "./isssample"
            out_name = line[0].split('.')[0]+'_'+line[1].split('.')[0].split('@')[1]+'_'+line[2]+".ply"
            ply_line_type = np.dtype(
                [
                    ("x", "f4"), 
                    ("y", "f4"),
                    ("z", "f4"), 
                    ("red", "u1"), 
                    ("green", "u1"), 
                    ("blue", "u1")
                    # ("nx", "f4"),
                    # ("ny", "f4"),
                    # ("nz", "f4")
                ]
            )
            utils.fuse2frags(points1, points2, ply_line_type, out_dir, out_name)
            utils.log_info(f"finish processing {out_name}")

