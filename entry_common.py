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
    
    register = models.registercore.RansacRegister(
        voxel_size=args.voxel_size,
        key_radius_factor=args.key_radius_factor,
        extracter_type=args.extracter_type,
        extracter_weights=args.state_dict,
        feat_radius_factor=args.voxel_size*2.0,
        feat_neighbour_num=50,
        ransac_workers_num=4,
        ransac_samples_num=4,
        ransac_corrdist_factor=2.0,
        ransac_iter_num=10000,
        ransac_vald_num=1000,
        ransac_rfne_num=25,
        checkr_corrdist_factor=2.0,
        checkr_mutldist_factor=0.85,
        checkr_normdegr_thresh=None
    )
    
    timer = utils.timer()
    for points1, points2, T_gdth, sample_name in dataloader:
        points1_o3d = utils.npy2o3d(points1)
        points1_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size*2.0, max_nn=30))
        points1 = utils.o3d2npy(points1_o3d)
        points2_o3d = utils.npy2o3d(points2)
        points2_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size*2.0, max_nn=30))
        points2 = utils.o3d2npy(points2_o3d)
        (
            fine_registrartion,
            downsampled_coords1, downsampled_coords2,
            keyptsdict1, keyptsdict2,
            totl_matches, gdth_matches
        ) = register.register(points1, points2, T_gdth)
        T_pred = fine_registrartion.transformation
        utils.log_info("pred T:", utils.resolve_axis_angle(T_pred, deg=True), T_pred[:3,3])
        utils.log_info("gdth T:", utils.resolve_axis_angle(T_gdth, deg=True), T_gdth[:3,3])
        
        utils.dump_registration_result(
            args.out_root, sample_name,
            points1, points2,
            downsampled_coords1, keyptsdict1["id"].values,
            downsampled_coords2, keyptsdict2["id"].values,
            T_gdth, T_pred,
            gdth_matches
        )

        utils.log_info(f"sample: {sample_name}")
