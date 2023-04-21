import os
import numpy as np
import open3d as o3d
from easydict import EasyDict as edict

from datasets import datasets
import config
import utils

import models


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
    
    # model configurations
    extracter_conf = edict({
        "extracter_type": args.extracter_type,
        # for fcgf
        "extracter_weight": args.extracter_weight,
        "fcgf_model": args.fcgf_model,
        # for fpfh
        "feat_radius": args.voxel_size * args.fpfh_radius_factor,
        "feat_neighbour_num": args.fpfh_nn
    })
    
    ransac_conf = edict({
        "num_workers": 4,
        "num_samples": 6,
        "max_corrdist": args.voxel_size * 1.5,
        "num_iter": 10000,
        "num_vald": 1000,
        "num_rfne": 25
    })
    
    checkr_conf = edict({
        "max_corrdist": args.voxel_size * 1.5,
        "mutldist_factor": 0.90,
        "normdegr_thresh": None
    })
    
    register = models.registercore.RansacRegister(
        voxel_size=args.voxel_size,
        # keypoint detector
        key_radius=args.voxel_size * args.key_radius_factor,
        # feature extracter
        extracter_conf=extracter_conf,
        # inlier proposal
        mapper_conf=args.mapper_conf,
        predictor_conf=args.predictor_conf,
        
        # optimization
        ransac_conf=ransac_conf,
        checkr_conf=checkr_conf
    )
    
    for points1, points2, T_gdth, sample_name in dataloader:
        points1_o3d = utils.npy2o3d(points1)
        points1_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2.0, max_nn=30))
        points1 = utils.o3d2npy(points1_o3d)
        points2_o3d = utils.npy2o3d(points2)
        points2_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2.0, max_nn=30))
        points2 = utils.o3d2npy(points2_o3d)
        (
            fine_registrartion,
            downsampled_coords1, downsampled_coords2,
            keyptsdict1, keyptsdict2,
            totl_matches, gdth_matches
        ) = register.register(points1, points2, T_gdth)
        if fine_registrartion is None:
            utils.log_warn(f"fail to register {sample_name}")
            continue
        
        T_pred = fine_registrartion.transformation
        utils.log_info("pred T:", utils.resolve_axis_angle(T_pred, deg=True), T_pred[:3,3])
        utils.log_info("gdth T:", utils.resolve_axis_angle(T_gdth, deg=True), T_gdth[:3,3])
        
        utils.dump_registration_result(
            args.out_root, "output",
            points1, points2,
            downsampled_coords1, keyptsdict1["id"].values,
            downsampled_coords2, keyptsdict2["id"].values,
            T_gdth, T_pred,
            gdth_matches
        )

        utils.log_info(f"sample: {sample_name}")
