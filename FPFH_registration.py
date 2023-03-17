import os
import numpy as np
import collections
import open3d as o3d

from datasets import datasets
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

    available_datasets = {attr_name: getattr(datasets, attr_name) for attr_name in dir(datasets) if callable(getattr(datasets, attr_name))}
    dataloader = available_datasets[args["data_type"]](
        root=args["data_root"],
        shuffle=True,
        augment=True,
        augdgre=30.0,
        augdist=2.0
    )

    for points1, points2, T_gdth, sample_name in dataloader:
        utils.log_info(sample_name)
        # step1: voxel down sample
        points1 = utils.voxel_down_sample(points1, args["ICP_radius"] * 2.0)
        points2 = utils.voxel_down_sample(points2, args["ICP_radius"] * 2.0)

        # step2: detect key points using ISS
        keypoints1 = utils.iss_detect(points1, args["ICP_radius"])
        keypoints2 = utils.iss_detect(points2, args["ICP_radius"])
        if len(keypoints1["id"].values) == 0 or len(keypoints2["id"].values) == 0:
            utils.log_warn(f"{sample_name} failed to find ISS keypoints, continue to next sample")
            continue

        # step3: compute FPFH for each key point
        # 一般原始点云旋转变换之后，再进行法向量估计好一些，万一出错呢。。。
        points1_o3d = utils.npy2o3d(points1)
        points1_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args["ICP_radius"]*2.0, max_nn=30))
        points2_o3d = utils.npy2o3d(points2)
        points2_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args["ICP_radius"]*2.0, max_nn=30))

        keyfpfhs1 = o3d.pipelines.registration.compute_fpfh_feature(
            points1_o3d.select_by_index(keypoints1["id"].values),
            o3d.geometry.KDTreeSearchParamHybrid(radius=args["ICP_radius"], max_nn=100)
        ).data
        keyfpfhs2 = o3d.pipelines.registration.compute_fpfh_feature(
            points2_o3d.select_by_index(keypoints2["id"].values),
            o3d.geometry.KDTreeSearchParamHybrid(radius=args["ICP_radius"], max_nn=100)
        ).data

        # # step4: ransac initial registration
        points1 = utils.o3d2npy(points1_o3d)
        points2 = utils.o3d2npy(points2_o3d)
        initial_ransac = utils.ransac_match(
            points1[keypoints1["id"].values], points2[keypoints2["id"].values],
            keyfpfhs1, keyfpfhs2,
            ransac_params=RANSACCONF(
                max_workers=4, num_samples=4,
                max_correspondence_dist=args["ICP_radius"] * 1.5,
                max_iter_num=100, max_valid_num=50, max_refine_num=30
            ),
            checkr_params=CHECKRCONF(
                max_correspondence_dist=args["ICP_radius"] * 1.5,
                max_mnn_dist_ratio=0.40,
                normal_angle_threshold=None
            )
        )

        # step5: ICP refinement
        dst_search_tree = o3d.geometry.KDTreeFlann(utils.npy2o3d(points2))
        final_result = utils.ICP_exact_match(
            points1, points2, dst_search_tree,
            initial_ransac.transformation,
            args["ICP_radius"], 1000
        )

        if len(final_result.correspondence_set) == 0:
            utils.log_warn(f"{sample_name} failed to recover the transformation")
            continue
        
        T_pred = final_result.transformation
        utils.log_info(f"pred T: {utils.resolve_axis_angle(T_pred, deg=True)}")
        utils.log_info(f"gdth T: {utils.resolve_axis_angle(T_gdth, deg=True)}")

        points1 = utils.apply_transformation(points1, T_pred)

        # output to file
        out_dir  = "./isssample"
        out_name = sample_name + ".ply"
        utils.fuse2frags(points1, points2, utils.ply_line_type, out_dir, out_name)
        utils.log_info(f"finish processing {out_name}")

