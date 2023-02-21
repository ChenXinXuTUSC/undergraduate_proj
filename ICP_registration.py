import os
import numpy as np
import collections
import open3d as o3d
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
            # , overwrite_rgb=True, new_rgb=[200, 0, 0]
            # , overwrite_rgb=True, new_rgb=[0, 200, 0]
            # step1: read point clouds from npz
            points1 = utils.npz2ply(os.path.join(args["3dmatch_root"], "npz", line[0]))
            points2 = utils.npz2ply(os.path.join(args["3dmatch_root"], "npz", line[1]))
            points1 = utils.voxel_down_sample(points1, args["ICP_radius"])
            points2 = utils.voxel_down_sample(points2, args["ICP_radius"])
            # randomly rotate and translate
            points1, rotmat, transd = utils.transform_augment(points1, 60.0, 2.0)

            # step2: detect key points using ISS
            keypoints1 = utils.iss_detect(points1, args["ICP_radius"])
            keypoints2 = utils.iss_detect(points2, args["ICP_radius"])

            # # step3: compute FPFH for each key point
            # 一般原始点云旋转变换之后，再进行法向量估计好一些，万一出错呢。。。
            points1_o3d = utils.npy2o3d(points1)
            points1_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args["ICP_radius"]*2.0, max_nn=30))
            points2_o3d = utils.npy2o3d(points2)
            points2_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args["ICP_radius"]*2.0, max_nn=30))

            keyfpfhs1 = o3d.pipelines.registration.compute_fpfh_feature(
                points1_o3d.select_by_index(keypoints1["id"].values),
                o3d.geometry.KDTreeSearchParamHybrid(radius=args["ICP_radius"]*2.0, max_nn=100)
            )
            keyfpfhs2 = o3d.pipelines.registration.compute_fpfh_feature(
                points2_o3d.select_by_index(keypoints2["id"].values),
                o3d.geometry.KDTreeSearchParamHybrid(radius=args["ICP_radius"]*2.0, max_nn=100)
            )

            regis_res = utils.ransac_match(
                points1, points2
            )
            points1[keypoints1["id"].values, 3:] = np.array([0, 255, 255])
            points2[keypoints2["id"].values, 3:] = np.array([0, 255, 255])

            out_dir  = "./isssample"
            # print(f"{line[0]}:{points1.shape}, {line[1]}:{points2.shape}, overlap ratio:{line[2]}")
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

