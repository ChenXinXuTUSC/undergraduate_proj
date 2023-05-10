import os
import numpy as np
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
# step4: feature description
# step5: RANSAC registration
#   step5.1: establish feature correspondences
#   step5.2: select n(> 3) pairs to solve the transformation R and t
#   step5.3: repeat step5.2 until error converge
# step6: ICP optimized transformation [R,t]

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
    feat_model = models.fcgf.load_model(args.feat_model)(
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
        points1_o3d = utils.npy2o3d(points1)
        points1_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size*2.0, max_nn=30))
        points1 = utils.o3d2npy(points1_o3d)
        points2_o3d = utils.npy2o3d(points2)
        points2_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size*2.0, max_nn=30))
        points2 = utils.o3d2npy(points2_o3d)

        # step1: voxel downsample
        downsampled_coords1, voxelized_coords1, idx_dse2vox1 = utils.voxel_down_sample_gpt(points1, args.voxel_size)
        downsampled_coords2, voxelized_coords2, idx_dse2vox2 = utils.voxel_down_sample_gpt(points2, args.voxel_size)

        # step2: detect key points using ISS
        keyptsdict1 = utils.iss_detect(downsampled_coords1, args.voxel_size * 0.95)
        keyptsdict2 = utils.iss_detect(downsampled_coords2, args.voxel_size * 0.95)
        keyptsindices1 = keyptsdict1["id"].values
        keyptsindices2 = keyptsdict2["id"].values
        if len(keyptsindices1) == 0 or len(keyptsindices2) == 0:
            utils.log_warn(f"{sample_name} failed to find ISS keypoints, continue to next sample")
            continue
        keypts1 = downsampled_coords1[keyptsindices1]
        keypts2 = downsampled_coords2[keyptsindices2]

        # step3: compute FCGF for each key point
        # compute all points' fcgf
        fcgfs1 = feat_model(
            ME.SparseTensor(
                coordinates=ME.utils.batched_coordinates([voxelized_coords1]).to(model_device), 
                features=torch.ones(len(downsampled_coords1), 1).to(model_device)
            )
        ).F.detach().cpu().numpy()
        fcgfs2 = feat_model(
            ME.SparseTensor(
                coordinates=ME.utils.batched_coordinates([voxelized_coords2]).to(model_device), 
                features=torch.ones(len(downsampled_coords2), 1).to(model_device)
            )
        ).F.detach().cpu().numpy()
        # only select key points' fcgf
        keyfcgfs1 = fcgfs1[keyptsindices1].T
        keyfcgfs2 = fcgfs2[keyptsindices2].T

        # step4: coarse ransac registration
        # use fpfh feature descriptor to compute matches
        matches = ransac.init_matches(keyfcgfs1, keyfcgfs2)
        correct = utils.ground_truth_matches(matches, keypts1, keypts2, args.voxel_size * 1.5, T_gdth) # 上帝视角
        correct_valid_num = correct.astype(np.int32).sum()
        correct_total_num = correct.shape[0]
        utils.log_info(f"gdth/init: {correct_valid_num:.2f}/{correct_total_num:.2f}={correct_valid_num/correct_total_num:.2f}")

        # 将对匹配对索引从关键点集合映射回原点云集合
        totl_matches = np.array([keyptsindices1[matches[:,0]], keyptsindices2[matches[:,1]]]).T
        gdth_matches = totl_matches[correct]

        initial_ransac = utils.ransac_match(
            keypts1, keypts2,
            keyfcgfs1, keyfcgfs2,
            ransac_params=edict({
                "max_workers":4, "num_samples":4,
                "max_corresponding_dist":args.voxel_size*2.0,
                "max_iter_num":2000, "max_valid_num":100, "max_refine_num":30
            }),
            checkr_params=edict({
                "max_corresponding_dist":args.voxel_size*2.0,
                "max_mnn_dist_ratio":0.85,
                "normal_angle_threshold":None
            }),
            matches=matches
        )

        if len(initial_ransac.correspondence_set) == 0:
            utils.log_warn(sample_name, "failed to recover the transformation")
            continue
        else:
            utils.log_info("correspondence set num:", len(initial_ransac.correspondence_set))
        
        final_result = icp.ICP_exact_match(
            downsampled_coords1, downsampled_coords2,
            o3d.geometry.KDTreeFlann(utils.npy2o3d(downsampled_coords2)), 
            initial_ransac.transformation, args.voxel_size,
            25
        )
        T_pred = final_result.transformation

        utils.log_info("pred T:", utils.resolve_axis_angle(T_pred, deg=True), T_pred[:3,3])
        utils.log_info("gdth T:", utils.resolve_axis_angle(T_gdth, deg=True), T_gdth[:3,3])
        utils.log_info(f"{sample_name}")
        # ============================= end of registration =============================



        # do some visualization works
        utils.dump_registration_result(
            args.out_root, sample_name,
            points1, points2,
            downsampled_coords1, keyptsindices1,
            downsampled_coords2, keyptsindices2,
            T_gdth, T_pred, gdth_matches
        )
        # break # for test
