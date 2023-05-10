import numpy as np
import open3d as o3d
from easydict import EasyDict as edict

from datasets import datasets
import config
import utils

import models

if __name__ == "__main__":
    args = config.args

    available_datasets = {attr_name: getattr(datasets, attr_name) for attr_name in dir(datasets) if callable(getattr(datasets, attr_name))}
    dataloader = available_datasets[args.data_type](
        root=args.data_root,
        shuffle=True,
        augdict= edict({
            "augment": True,
            "augdgre": 90.0,
            "augdist": 5.0,
            "augjitr": 0.00,
            "augnois": 0
        }),
        args=args
    )
    
    # model configurations
    detecter_conf = edict({
        "key_radius": args.voxel_size * args.key_radius_factor,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2
    })
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
        "num_samples": 8,
        "max_corrdist": args.voxel_size * 1.25,
        "num_iter": 7500,
        "num_vald": 750,
        "num_rfne": 25
    })
    checkr_conf = edict({
        "max_corrdist": args.voxel_size * 1.25,
        "mutldist_factor": 0.85,
        "normdegr_thresh": None
    })
    
    register = models.registercore.RansacRegister(
        voxel_size=args.voxel_size,
        # keypoint detector
        detecter_conf=detecter_conf,
        # feature extracter
        extracter_conf=extracter_conf,
        # inlier proposal
        mapper_conf=args.mapper_conf,
        predicter_conf=args.predicter_conf,
        
        # optimization
        ransac_conf=ransac_conf,
        checkr_conf=checkr_conf,
        
        misc=args
    )
    
    for points1, points2, T_gdth, sample_name in dataloader:
        utils.log_info(sample_name)
        if args.recompute_norm:
            points1_o3d = utils.npy2o3d(points1)
            points1_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2.0, max_nn=50))
            points1 = utils.o3d2npy(points1_o3d)
            points2_o3d = utils.npy2o3d(points2)
            points2_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2.0, max_nn=50))
            points2 = utils.o3d2npy(points2_o3d)
        
        # voxel domnsample
        downsampled_coords, voxelized_coords, idx_dse2vox = register.downsample(points1)
        downsampled_coords[:, 3:6] = np.array([100, 100, 100])
        # detect keypoints
        keyptsidx, *misc = register.detect_keypoints(downsampled_coords, add_salt=False)

        # compute features
        feats = register.extract_features(downsampled_coords, voxelized_coords)
        
        selected_idx = np.random.choice(keyptsidx, size=3, replace=False)
        # color query point
        colors = [
            np.array([255, 255, 0]),
            np.array([255, 0, 255]),
            np.array([0, 255, 255])
        ]
        
        # FLANN search
        search_tree = o3d.geometry.KDTreeFlann(feats.T)
        neighbours = []
        for i, idx in enumerate(selected_idx):
            _, dst_idx, _ = search_tree.search_knn_vector_xd(feats[idx], 250)
            neighbours.append(downsampled_coords[dst_idx])
            neighbours[-1][:, 3:6] = colors[i]
        neighbours = np.concatenate(neighbours, axis=0)
        
        utils.dump1frag(downsampled_coords, utils.make_ply_vtx_type(True, True), out_dir=args.out_root, out_name="original.ply")
        utils.dump1frag(neighbours, utils.make_ply_vtx_type(True, True), out_dir=args.out_root, out_name="neighbours.ply")
