import os
import numpy as np
import collections
import open3d as o3d

from datasets import datasets
import config
import utils

from utils import ransac
from utils import icp

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

    available_datasets = {attr_name: getattr(datasets, attr_name) for attr_name in dir(datasets) if callable(getattr(datasets, attr_name))}
    dataloader = available_datasets[args["data_type"]](
        root=args["data_root"],
        shuffle=True,
        augment=True,
        augdgre=30.0,
        augdist=4.0
    )

    for points1, points2, T, sample_name in dataloader:
        utils.log_info(sample_name)
        # step1: voxel downsample
        points1 = utils.voxel_down_sample(points1, args["ICP_radius"])
        points2 = utils.voxel_down_sample(points2, args["ICP_radius"])

        # augment information
        rotmat, transd = T[:3,:3], T[:3,3]
        raxis, angle = utils.resolve_axis_angle(rotmat)
        utils.log_info(f"augment raxis:{raxis}, angle:{np.arctan(angle)*180/np.pi :.2f}, transd:{transd}")

        # step2: detect key points using ISS
        keypoints1 = utils.iss_detect(points1, args["ICP_radius"])
        keypoints2 = utils.iss_detect(points2, args["ICP_radius"])
        if len(keypoints1["id"].values) == 0 or len(keypoints2["id"].values) == 0:
            utils.log_warn(f"{sample_name} failed to find ISS keypoints, continue to next sample")
            continue
        # 给关键点上亮色
        points1[keypoints1["id"].values, 3:6] = np.array([0, 255, 255])
        points2[keypoints2["id"].values, 3:6] = np.array([0, 255, 255])

        # step3: compute FPFH for each key point
        points1_o3d = utils.npy2o3d(points1)
        points1_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args["ICP_radius"]*2.0, max_nn=30))
        points2_o3d = utils.npy2o3d(points2)
        points2_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args["ICP_radius"]*2.0, max_nn=30))

        points1_kps = points1_o3d.select_by_index(keypoints1["id"].values)
        keyfpfhs1 = o3d.pipelines.registration.compute_fpfh_feature(
            points1_kps,
            o3d.geometry.KDTreeSearchParamHybrid(radius=args["ICP_radius"]*5.0, max_nn=100)
        ).data
        points2_kps = points2_o3d.select_by_index(keypoints2["id"].values)
        keyfpfhs2 = o3d.pipelines.registration.compute_fpfh_feature(
            points2_kps,
            o3d.geometry.KDTreeSearchParamHybrid(radius=args["ICP_radius"]*5.0, max_nn=100)
        ).data

        # 给关键点做FPFH近邻搜索匹配画线
        matches = ransac.init_matches(keyfpfhs1, keyfpfhs2)
        matches = np.array([keypoints1["id"].values[matches[:,0]], keypoints2["id"].values[matches[:,1]]]).T

        # step4: ransac initial registration
        initial_ransac = utils.ransac_match_copy(
            None,        None,
            points1_kps, points2_kps,
            keyfpfhs1,   keyfpfhs2,
            ransac_params=RANSACCONF(
                max_workers=4, num_samples=4,
                max_correspondence_dist=args["ICP_radius"]*1.5,
                max_iter_num=20000, max_valid_num=500, max_refine_num=30
            ),
            checkr_params=CHECKRCONF(
                max_correspondence_dist=args["ICP_radius"]*1.5,
                max_mnn_dist_ratio=0.81,
                normal_angle_threshold=None
            )
        )

        if len(initial_ransac.correspondence_set) == 0:
            utils.log_warn(f"{sample_name} failed to recover the transformation")
            continue
        
        search_tree_points2 = o3d.geometry.KDTreeFlann(points2_o3d)
        final_result = icp.ICP_exact_match_copy(
            points1_o3d, points2_o3d, search_tree_points2, 
            initial_ransac.transformation, args["ICP_radius"]*1.0,
            1000
        )

        T_pred = final_result.transformation
        rotmat_pred = T_pred[:3, :3]
        transd_pred = T_pred[:3, 3]
        raxis, angle = utils.resolve_axis_angle(rotmat_pred)
        utils.log_info(f"pred raxis:{raxis}, angle:{np.arctan(angle)*180/np.pi :.2f}, transd:{transd_pred}")

        points1 = utils.apply_transformation(points1, T_pred)

        # output to file
        out_dir  = "./samples/matches_sample"
        out_name = sample_name + ".ply"
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
        ply_edge_type = np.dtype(
            [
                ("vertex1", "uint32"), 
                ("vertex2", "uint32"),
                ("red", "u1"), 
                ("green", "u1"), 
                ("blue", "u1")
            ]
        )
        utils.fuse2frags_with_matches(points1, points2, matches, ply_vertex_type, ply_edge_type, out_dir, out_name)
        utils.log_info(f"finish processing {out_name}")

