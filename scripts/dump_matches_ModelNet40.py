import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)) , ".."))

import numpy as np
import open3d as o3d

from tqdm import tqdm
from easydict import EasyDict as edict

from datasets import datasets
import config
import utils
from utils import ransac

def add_salt(total: int, selected: np.ndarray, noise_ratio: float):
    if noise_ratio < 1e-5:
        return selected
    fullset = set(list(range(total)))
    subbset = set(list(selected))
    rndptsidx = np.random.choice(list(fullset - subbset), size=int(total * noise_ratio), replace=False)
    return np.concatenate([selected, rndptsidx])

if __name__ == "__main__":
    args = edict(vars(config.args))

    available_datasets = {attr_name: getattr(datasets, attr_name) for attr_name in dir(datasets) if callable(getattr(datasets, attr_name))}
    dataloader = available_datasets[args.data_type](
        root=args.data_root,
        shuffle=True,
        augdict= edict({
            "augment": False,
            "augdgre": 90.0,
            "augdist": 5.0,
            "augjitr": 0.00,
            "augnois": 0
        }),
        args=args
    )

    if not os.path.exists(args.out_root):
        os.makedirs(args.out_root, mode=0o755)

    for i, (points1, points2, T_gdth, sample_name) in tqdm(enumerate(dataloader), total=len(dataloader), ncols=100, desc="gen npz"):
        if args.recompute_norm:
            points1_o3d = utils.npy2o3d(points1)
            points1_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2.0, max_nn=50))
            points1 = utils.o3d2npy(points1_o3d)
            points2_o3d = utils.npy2o3d(points2)
            points2_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2.0, max_nn=50))
            points2 = utils.o3d2npy(points2_o3d)
        # step1: voxel downsample
        downsampled_coords1, voxelized_coords1, idx_dse2vox1 = utils.voxel_down_sample_gpt(points1, args.voxel_size)
        downsampled_coords2, voxelized_coords2, idx_dse2vox2 = utils.voxel_down_sample_gpt(points2, args.voxel_size)

        # step2: detect key points using random selection
        keyptsdict1 = utils.iss_detect(
            downsampled_coords1,
            args.voxel_size * args.key_radius_factor,
            args.lambda1, args.lambda2
        )
        keyptsdict2 = utils.iss_detect(
            downsampled_coords2,
            args.voxel_size * args.key_radius_factor,
            args.lambda1, args.lambda2
        )
        keyptsidx1 = keyptsdict1["id"].values
        keyptsidx2 = keyptsdict2["id"].values
        if len(keyptsidx1) == 0 or len(keyptsidx2) == 0:
            utils.log_warn(f"{sample_name} failed to find ISS keypoints, continue to next sample")
            continue
        keyptsidx1 = add_salt(len(downsampled_coords1), keyptsidx1, args.salt_keypts)
        keyptsidx2 = add_salt(len(downsampled_coords2), keyptsidx2, args.salt_keypts)

        # step3: compute FPFH for each key point
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
        correct = utils.ground_truth_matches(
            matches,
            downsampled_coords1[keyptsidx1],
            downsampled_coords2[keyptsidx2],
            args.voxel_size * 1.5, T_gdth
        )
        num_valid_matches = int(correct.astype(np.int32).sum())
        num_total_matches = int(correct.shape[0])
        tqdm.write(utils.log_info(f"gdth/init: {num_valid_matches:4d}/{num_total_matches:4d}={num_valid_matches/num_total_matches:.2f}", quiet=True))
        
        matches_mat = np.concatenate([keyfeats1[matches[:, 0]], keyfeats2[matches[:, 1]]], axis=1)
        
        np.savez(f"{args.out_root}/{sample_name}", features=matches_mat, labels=correct)
