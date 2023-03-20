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
import models

import MinkowskiEngine as ME
import torch

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


def preprocess(points:np.ndarray, voxel_size:float):
    voxcoords, voxidxs = ME.utils.sparse_quantize(
        coordinates=points[:,:3], 
        quantization_size=voxel_size,
        return_index=True
    )
    voxcoords = ME.utils.batched_coordinates([voxcoords])
    feats = torch.ones(len(voxidxs), 1)
    return torch.from_numpy(points[voxidxs, :3]).float(), voxcoords, voxidxs, feats


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

    model_conf = torch.load(args.state_dict)["config"]
    model_params = torch.load(args.state_dict)["state_dict"]
    feat_model = models.load_model(args.feat_model)(
        1,
        model_conf["model_n_out"],
        bn_momentum=model_conf["bn_momentum"],
        conv1_kernel_size=model_conf['conv1_kernel_size'],
        normalize_feature=model_conf['normalize_feature']
    )
    feat_model.load_state_dict(model_params)
    feat_model.eval()


    for points1, points2, T_gdth, sample_name in dataloader:
        utils.log_info(sample_name)
        # step1: voxel downsample
        xyzs1, voxcoords1, voxidxs1, feats1 = preprocess(points1, args.ICP_radius)
        xyzs2, voxcoords2, voxidxs2, feats2 = preprocess(points2, args.ICP_radius)

        # step2: detect key points using ISS
        keyptsdict1 = utils.iss_detect(xyzs1, args.ICP_radius)
        keyptsdict2 = utils.iss_detect(xyzs2, args.ICP_radius)
        if len(keyptsdict1["id"].values) == 0 or len(keyptsdict2["id"].values) == 0:
            utils.log_warn(f"{sample_name} failed to find ISS keypoints, continue to next sample")
            continue
        keypts1 = xyzs1[keyptsdict1["id"].values]
        keypts2 = xyzs2[keyptsdict2["id"].values]

        # step3: compute FCGF for each key point
        # compute all points' fcgf
        fcgfs1 = feat_model(ME.SparseTensor(coordinates=voxcoords1, features=feats1)).F
        fcgfs2 = feat_model(ME.SparseTensor(coordinates=voxcoords2, features=feats2)).F
        # only select key points' fcgf
        keyfcgfs1 = fcgfs1[keyptsdict1["id"].values]
        keyfcgfs2 = fcgfs1[keyptsdict2["id"].values]

        # use fpfh feature descriptor to compute matches
        matches = ransac.init_matches(keyfcgfs1, keyfcgfs2)
        correct = utils.ground_truth_matches(matches, keypts1, keypts2, args.ICP_radius * 2.5, T_gdth) # 上帝视角
        utils.log_info("gdth matches:", correct.astype(np.int32).sum())
        # 将对匹配对索引从关键点集合映射回原点云集合
        init_matches = np.array([keyptsdict1["id"].values[matches[:,0]], keyptsdict2["id"].values[matches[:,1]]]).T
        gdth_matches = np.array([keyptsdict1["id"].values[matches[:,0]], keyptsdict2["id"].values[matches[:,1]]]).T[correct]

        # step4: ransac initial registration
        initial_ransac = utils.ransac_match(
            keypts1, keypts2,
            keyfcgfs1, keyfcgfs2,
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
        
        search_tree_points2 = o3d.geometry.KDTreeFlann(xyzs2)
        final_result = icp.ICP_exact_match(
            xyzs1, xyzs2, search_tree_points2, 
            initial_ransac.transformation, args.ICP_radius,
            100
        )

        T_pred = final_result.transformation
        utils.log_info("pred T:", utils.resolve_axis_angle(T_pred, deg=True), T_pred[:3,3])
        utils.log_info("gdth T:", utils.resolve_axis_angle(T_gdth, deg=True), T_gdth[:3,3])
        xyzs1 = utils.apply_transformation(xyzs1, T_pred)

        # output to file
        xyzs1_o3d = o3d.geometry.PointCloud()
        xyzs1_o3d.points = o3d.utility.Vector3dVector(xyzs1)
        xyzs2_o3d = o3d.geometry.PointCloud()
        xyzs2_o3d.points = o3d.utility.Vector3dVector(xyzs2)
        
        # 给关键点上亮色，请放在其他点上色完成后再给关键点上色，否则关键点颜色会被覆盖
        points1[keyptsdict1["id"].values, 3:6] = np.array([255, 0, 0])
        points2[keyptsdict2["id"].values, 3:6] = np.array([0, 255, 0])

        out_dir  = "./samples/matches_sample"
        utils.fuse2frags_with_matches(points1, points2, init_matches, utils.ply_vertex_type, utils.ply_edge_type, out_dir, "init_matches.ply")
        utils.fuse2frags_with_matches(points1, points2, gdth_matches, utils.ply_vertex_type, utils.ply_edge_type, out_dir, "gdth_matches.ply")
        utils.log_info(f"finish processing {sample_name}")
        break # only for test

