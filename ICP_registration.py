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

#RANSAC configuration:
RANSACCONF = collections.namedtuple(
    "RANSACCONF",
    [
        "max_workers",
        "num_samples",
        "max_correspondence_dist", 'max_iter_num', 'max_valid_num', 'max_refine_num'
    ]
)
# fast pruning algorithm configuration:
CHECKRCONF = collections.namedtuple(
    "CHECKRCONF",
    [
        "max_correspondence_dist",
        "max_mnn_dist_ratio", 
        "normal_angle_threshold"
    ]
)

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
            ).data
            keyfpfhs2 = o3d.pipelines.registration.compute_fpfh_feature(
                points2_o3d.select_by_index(keypoints2["id"].values),
                o3d.geometry.KDTreeSearchParamHybrid(radius=args["ICP_radius"]*2.0, max_nn=100)
            ).data

            regis_res = utils.ransac_match(
                utils.o3d2npy(points1_o3d), utils.o3d2npy(points2_o3d),
                keyfpfhs1, keyfpfhs2,
                ransac_params=RANSACCONF(
                    max_workers=4, num_samples=4,
                    max_correspondence_dist=0.20,
                    max_iter_num=100, max_valid_num=50, max_refine_num=100
                ),
                checkr_params=CHECKRCONF(
                    max_correspondence_dist=0.20,
                    max_mnn_dist_ratio=0.8,
                    normal_angle_threshold=None
                )
            )

            rotmat_pred = regis_res.transform[:3, :3]
            transd_pred = regis_res.transform[:3, 3]
            R_err = ((rotmat_pred-rotmat)*(rotmat_pred-rotmat)).sum()
            t_err = ((transd_pred-transd)*(transd_pred-transd)).sum()
            utils.log_info(f"R err: {R_err:.3f}, t err: {t_err:.3f}")
            input("press any key to continue...")

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

