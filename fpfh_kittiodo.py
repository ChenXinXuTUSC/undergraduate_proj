import os
import numpy as np
import collections
import open3d as o3d
from easydict import EasyDict as edict

from datasets import datasets
import config
import utils

from utils import ransac
from utils import icp

# step1: read point cloud pair
# step2: voxel down sample
# step3: extract ISS feature
# step4: feature description
# step5: RANSAC registration
#   step5.1: establish feature correspondences
#   step5.2: select n(> 3) pairs to solve the transformation R and t
#   step5.3: repeat step5.2 until error converge
# step6: ICP optimized transformation [R,t]

#RANSAC configuration:
RANSACCONF = collections.namedtuple(
    "RANSACCONF",
    [
        "max_workers",
        "num_samples",
        "max_corresponding_dist", 'max_iter_num', 'max_valid_num', 'max_refine_num'
    ]
)
# fast pruning algorithm configuration:
CHECKRCONF = collections.namedtuple(
    "CHECKRCONF",
    [
        "max_corresponding_dist",
        "max_mnn_dist_ratio", 
        "normal_angle_threshold"
    ]
)

if __name__ == "__main__":
    args = edict(vars(config.args))

    available_datasets = {attr_name: getattr(datasets, attr_name) for attr_name in dir(datasets) if callable(getattr(datasets, attr_name))}
    dataloader = available_datasets[args.data_type](
        root=args.data_root,
        shuffle=True,
        augment=True,
        augdgre=30.0,
        augdist=4.0,
        args=args
    )

    for points1, points2, T_gdth, sample_name in dataloader:
        utils.log_info(sample_name)
        # step1: voxel downsample
        # already voxel downsampled in dataloader

        # step2: detect key points using ISS
        keyptsdict1 = utils.iss_detect(points1, args.ICP_radius)
        keyptsdict2 = utils.iss_detect(points2, args.ICP_radius)
        if len(keyptsdict1["id"].values) == 0 or len(keyptsdict2["id"].values) == 0:
            utils.log_warn(f"{sample_name} failed to find ISS keypoints, continue to next sample")
            continue
        keypts1 = points1[keyptsdict1["id"].values]
        keypts2 = points2[keyptsdict2["id"].values]

        # step3: compute FPFH for each key point
        points1_o3d = utils.npy2o3d(points1)
        points2_o3d = utils.npy2o3d(points2)
        points1_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.ICP_radius*2.0, max_nn=50))
        points2_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.ICP_radius*2.0, max_nn=50))
        # compute all points' fpfh
        fpfhs1 = o3d.pipelines.registration.compute_fpfh_feature(
            points1_o3d,
            o3d.geometry.KDTreeSearchParamHybrid(radius=args.ICP_radius*5.0, max_nn=100)
        ).data
        fpfhs2 = o3d.pipelines.registration.compute_fpfh_feature(
            points2_o3d,
            o3d.geometry.KDTreeSearchParamHybrid(radius=args.ICP_radius*5.0, max_nn=100)
        ).data
        # only select key points' fpfh
        keyfpfhs1 = fpfhs1[:, keyptsdict1["id"].values]
        keyfpfhs2 = fpfhs2[:, keyptsdict2["id"].values]

        # use fpfh feature descriptor to compute matches
        matches = ransac.init_matches(keyfpfhs1, keyfpfhs2)
        correct = utils.ground_truth_matches(matches, keypts1, keypts2, args.ICP_radius * 2.5, T_gdth) # 上帝视角
        utils.log_info("gdth matches:", correct.astype(np.int32).sum())
        # 将对匹配对索引从关键点集合映射回原点云集合
        init_matches = np.array([keyptsdict1["id"].values[matches[:,0]], keyptsdict2["id"].values[matches[:,1]]]).T
        gdth_matches = np.array([keyptsdict1["id"].values[matches[:,0]], keyptsdict2["id"].values[matches[:,1]]]).T[correct]

        # step4: ransac initial registration
        initial_ransac = utils.ransac_match(
            keypts1, keypts2,
            keyfpfhs1, keyfpfhs2,
            ransac_params=RANSACCONF(
                max_workers=4, num_samples=4,
                max_corresponding_dist=args.ICP_radius*2.5,
                max_iter_num=2000, max_valid_num=100, max_refine_num=30
            ),
            checkr_params=CHECKRCONF(
                max_corresponding_dist=args.ICP_radius*2.5,
                max_mnn_dist_ratio=0.50,
                normal_angle_threshold=None
            ),
            matches=matches
        )

        if len(initial_ransac.correspondence_set) == 0:
            utils.log_warn(sample_name, "failed to recover the transformation")
            continue
        
        search_tree_points2 = o3d.geometry.KDTreeFlann(points2_o3d)
        final_result = icp.ICP_exact_match(
            points1, points2, search_tree_points2, 
            initial_ransac.transformation, args.ICP_radius,
            100
        )

        T_pred = final_result.transformation
        utils.log_info("pred T:", utils.resolve_axis_angle(T_pred, deg=True), T_pred[:3,3])
        utils.log_info("gdth T:", utils.resolve_axis_angle(T_gdth, deg=True), T_gdth[:3,3])
        points1 = utils.apply_transformation(points1, T_pred)

        # output to file
        points1[:, 6:9] = np.asarray(points1_o3d.normals)
        points2[:, 6:9] = np.asarray(points2_o3d.normals)
        points1[:, 3:6] = np.array([200, 200, 0], dtype=np.int32)
        points2[:, 3:6] = np.array([0, 200, 200], dtype=np.int32)
        # 给关键点上亮色，请放在其他点上色完成后再给关键点上色，否则关键点颜色会被覆盖
        points1[keyptsdict1["id"].values, 3:6] = np.array([255, 0, 0])
        points2[keyptsdict2["id"].values, 3:6] = np.array([0, 255, 0])

        utils.fuse2frags_with_matches(points1, points2, init_matches, utils.ply_vertex_type, utils.ply_edge_type, args.out_root, "init_matches.ply")
        utils.fuse2frags_with_matches(points1, points2, gdth_matches, utils.ply_vertex_type, utils.ply_edge_type, args.out_root, "gdth_matches.ply")
        utils.log_info(f"finish processing {sample_name}")
        break # only for test

