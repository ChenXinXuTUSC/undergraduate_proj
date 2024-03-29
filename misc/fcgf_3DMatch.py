import numpy as np
import collections
import open3d as o3d
from easydict import EasyDict as edict

import time

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

@torch.no_grad()
def preprocess(points:np.ndarray, voxel_size:float):
    voxcoords, voxidxs = ME.utils.sparse_quantize(
        coordinates=points[:,:3], 
        quantization_size=voxel_size,
        return_index=True
    )
    voxcoords = ME.utils.batched_coordinates([voxcoords])
    feats = torch.ones(len(voxidxs), 1)
    return points[voxidxs], voxcoords, voxidxs, feats


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
        conv1_kernel_size=model_conf["conv1_kernel_size"],
        normalize_feature=model_conf["normalize_feature"]
    )
    feat_model.load_state_dict(model_params)
    feat_model.eval()
    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_model.to(model_device)

    timer = utils.timer()
    for points1, points2, T_gdth, sample_name in dataloader:
        utils.log_info(f"processing {sample_name}")
        # FCGF model needs:
        # 1. down sampled points' coordinates
        # 2. voxielized points' coordinates
        
        # step1: voxel downsample
        xyzs1, voxcoords1, voxidxs1, feats1 = preprocess(points1, args.ICP_radius)
        xyzs2, voxcoords2, voxidxs2, feats2 = preprocess(points2, args.ICP_radius)

        xyzs1_o3d = utils.npy2o3d(xyzs1)
        xyzs2_o3d = utils.npy2o3d(xyzs2)

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
        fcgfs1 = feat_model(ME.SparseTensor(coordinates=voxcoords1.to(model_device), features=feats1.to(model_device))).F.detach().cpu().numpy()
        fcgfs2 = feat_model(ME.SparseTensor(coordinates=voxcoords2.to(model_device), features=feats2.to(model_device))).F.detach().cpu().numpy()
        # only select key points' fcgf
        keyfcgfs1 = fcgfs1[keyptsdict1["id"].values].T
        keyfcgfs2 = fcgfs2[keyptsdict2["id"].values].T

        # use fpfh feature descriptor to compute matches
        matches = ransac.init_matches(keyfcgfs1, keyfcgfs2)
        correct = utils.ground_truth_matches(matches, keypts1, keypts2, args.ICP_radius * 2.5, T_gdth) # 上帝视角
        correct_valid_num = correct.astype(np.int32).sum()
        correct_total_num = correct.shape[0]
        utils.log_info(f"gdth/init: {correct_valid_num:.2f}/{correct_total_num:.2f}={correct_valid_num/correct_total_num:.2f}")

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
                max_mnn_dist_ratio=0.80,
                normal_angle_threshold=None
            )
        )

        if len(initial_ransac.correspondence_set) == 0:
            utils.log_warn(sample_name, "failed to recover the transformation")
            continue
        else:
            utils.log_info("correspondence set num:", len(initial_ransac.correspondence_set))
        
        search_tree_points2 = o3d.geometry.KDTreeFlann(xyzs2_o3d)
        final_result = icp.ICP_exact_match(
            xyzs1, xyzs2, search_tree_points2, 
            initial_ransac.transformation, args.ICP_radius,
            100
        )

        T_pred = final_result.transformation
        utils.log_info("pred T:", utils.resolve_axis_angle(T_pred, deg=True), T_pred[:3,3])
        utils.log_info("gdth T:", utils.resolve_axis_angle(T_gdth, deg=True), T_gdth[:3,3])
        xyzs1 = utils.apply_transformation(xyzs1, T_pred)

        # 给关键点上亮色，请放在其他点上色完成后再给关键点上色，否则关键点颜色会被覆盖
        xyzs1[keyptsdict1["id"].values, 3:6] = np.array([255, 0, 0])
        xyzs2[keyptsdict2["id"].values, 3:6] = np.array([0, 255, 0])

        # output to file
        utils.fuse2frags_with_matches(xyzs1, xyzs2, init_matches, utils.make_ply_vtx_type(True, True), utils.ply_edg_i1i2rgb, args.out_root, "init_matches.ply")
        utils.fuse2frags_with_matches(xyzs1, xyzs2, gdth_matches, utils.make_ply_vtx_type(True, True), utils.ply_edg_i1i2rgb, args.out_root, "gdth_matches.ply")
        points1_o3d = utils.npy2o3d(points1)
        points2_o3d = utils.npy2o3d(points2)
        points1_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.ICP_radius*2.0, max_nn=50))
        points2_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.ICP_radius*2.0, max_nn=50))
        utils.fuse2frags(
            utils.apply_transformation(utils.o3d2npy(points1_o3d), T_pred),
            utils.o3d2npy(points2_o3d), 
            utils.make_ply_vtx_type(True, True), args.out_root, "original_color.ply"
        )
        points1_o3d.paint_uniform_color([1.0, 1.0, 0.0])
        points2_o3d.paint_uniform_color([0.0, 1.0, 1.0])
        points1 = utils.o3d2npy(points1_o3d)
        points2 = utils.o3d2npy(points2_o3d)
        points1[:,3:6] = points1[:,3:6] * 255.0
        points2[:,3:6] = points2[:,3:6] * 255.0
        utils.fuse2frags(
            utils.apply_transformation(points1, T_pred),
            points2, 
            utils.make_ply_vtx_type(True, True), args.out_root, "pred.ply"
        )
        utils.fuse2frags(
            utils.apply_transformation(points1, T_gdth),
            points2, 
            utils.make_ply_vtx_type(True, True), args.out_root, "gdth.ply"
        )
        utils.log_info(f"finish processing {sample_name}")
        break # only for test

