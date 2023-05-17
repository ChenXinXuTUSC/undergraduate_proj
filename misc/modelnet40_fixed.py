import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)) , ".."))

import numpy as np
import open3d as o3d

from tqdm import tqdm
from easydict import EasyDict as edict

import models
from datasets import datasets
import config
import utils
from utils import ransac


if __name__ == "__main__":
    args = edict(vars(config.args))

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
        "max_corrdist": args.voxel_size * 1.50,
        "num_iter": 7500,
        "num_vald": 750,
        "num_rfne": 25
    })
    checkr_conf = edict({
        "max_corrdist": args.voxel_size * 1.50,
        "mutldist_factor": 0.90,
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

    if not os.path.exists(args.out_root):
        os.makedirs(args.out_root, mode=0o755)

    for points1, points2, T_gdth, sample_name in dataloader:
        utils.log_info(sample_name)
        
        if args.recompute_norm:
            points1_o3d = utils.npy2o3d(points1)
            points1_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2.0, max_nn=50))
            points1 = utils.o3d2npy(points1_o3d)
            points2_o3d = utils.npy2o3d(points2)
            points2_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2.0, max_nn=50))
            points2 = utils.o3d2npy(points2_o3d)
        # step1: voxel downsample
        downsampled_coords1, voxelized_coords1, idx_dse2vox1 = register.downsample(points1)
        downsampled_coords2, voxelized_coords2, idx_dse2vox2 = register.downsample(points2)

        # step2: detect key points using random selection
        keyptsidx1, *_ = register.detect_keypoints(downsampled_coords1, args.salt_keypts)
        keyptsidx2, *_ = register.detect_keypoints(downsampled_coords2, args.salt_keypts)
        if len(keyptsidx1) == 0 or len(keyptsidx2) == 0:
            utils.log_warn(f"{sample_name} failed to find ISS keypoints, continue to next sample")
            continue
        keypts1 = downsampled_coords1[keyptsidx1]
        keypts2 = downsampled_coords2[keyptsidx2]

        # step3: compute FPFH for each key point
        # compute all points' fpfh
        fpfhs1 = o3d.pipelines.registration.compute_fpfh_feature(
            utils.npy2o3d(downsampled_coords1),
            o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * args.fpfh_radius_factor, max_nn=args.fpfh_nn)
        ).data.T
        fpfhs2 = o3d.pipelines.registration.compute_fpfh_feature(
            utils.npy2o3d(downsampled_coords2),
            o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * args.fpfh_radius_factor, max_nn=args.fpfh_nn)
        ).data.T
        # only select key points' fpfh
        keyfeats1 = fpfhs1[keyptsidx1]
        keyfeats2 = fpfhs2[keyptsidx2]

        # step4: coarse ransac registration
        # use fpfh feature descriptor to compute matches
        matches = ransac.init_matches(keyfeats1.T, keyfeats2.T)
        (
            coarse_registrartion,
            totl_matches, gdth_matches
        ) = register.coarse_registration(
            downsampled_coords1, downsampled_coords2,
            keyptsidx1, keyptsidx2,
            fpfhs1, fpfhs2,
            T_gdth, matches
        )
        if coarse_registrartion is None:
            utils.log_warn(f"failed to register {sample_name}")
            continue
        
        T_pred = coarse_registrartion.transformation
        raxis_pred, rdegr_pred = utils.resolve_axis_angle(T_pred, deg=True)
        raxis_gdth, rdegr_gdth = utils.resolve_axis_angle(T_gdth, deg=True)
        trans_pred, trans_gdth = T_pred[:3,3], T_gdth[:3,3]
        print(utils.get_colorstr(
                fore=utils.FORE_CYN, back=utils.BACK_ORG,
                msg="raxis\trdegr\ttrans"
            )
        )
        print(utils.get_colorstr(
                fore=utils.FORE_PRP, back=utils.BACK_ORG,
                msg=f"{np.arccos(np.dot(raxis_gdth, raxis_pred)):5.3f}\t{abs(rdegr_gdth - rdegr_pred):5.3f}\t{[float(f'{x:.2f}') for x in trans_gdth - trans_pred]}"
            )
        )
        
        utils.dump_registration_result(
            args.out_root,
            "output",
            points1, points2,
            downsampled_coords1, keyptsidx1,
            downsampled_coords2, keyptsidx2,
            T_gdth, T_pred,
            gdth_matches
        )
