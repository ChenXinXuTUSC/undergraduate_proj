import os
import numpy as np
import open3d as o3d

from tqdm import tqdm
from easydict import EasyDict as edict

from datasets import datasets
import config
import utils
from utils import ransac


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

    if not os.path.exists(args.out_root):
        os.makedirs(args.out_root, mode=0o755)

    for i, (points1, points2, T_gdth, sample_name) in tqdm(enumerate(dataloader), total=len(dataloader), ncols=100, desc="gen npz"):
        # step1: voxel downsample
        downsampled_coords1, voxelized_coords1, idx_dse2vox1 = utils.voxel_down_sample_gpt(points1, args.voxel_size)
        downsampled_coords2, voxelized_coords2, idx_dse2vox2 = utils.voxel_down_sample_gpt(points2, args.voxel_size)

        # step2: detect key points using random selection
        keyptsdict1 = utils.iss_detect(downsampled_coords1, args.voxel_size * 2.0)
        keyptsdict2 = utils.iss_detect(downsampled_coords2, args.voxel_size * 2.0)
        if len(keyptsdict1["id"].values) == 0 or len(keyptsdict2["id"].values) == 0:
            utils.log_warn(f"{sample_name} failed to find ISS keypoints, continue to next sample")
            continue
        keyptsindices1 = keyptsdict1["id"].values
        keyptsindices2 = keyptsdict2["id"].values
        keypts1 = downsampled_coords1[keyptsindices1]
        keypts2 = downsampled_coords2[keyptsindices2]

        # step3: compute FPFH for each key point
        downsampled_coords1_o3d = utils.npy2o3d(downsampled_coords1)
        downsampled_coords2_o3d = utils.npy2o3d(downsampled_coords2)
        downsampled_coords1_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size*2.0, max_nn=50))
        downsampled_coords2_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size*2.0, max_nn=50))
        # compute all points' fpfh
        fpfhs1 = o3d.pipelines.registration.compute_fpfh_feature(
            downsampled_coords1_o3d,
            o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size*1.5, max_nn=100)
        ).data.T
        fpfhs2 = o3d.pipelines.registration.compute_fpfh_feature(
            downsampled_coords2_o3d,
            o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size*1.5, max_nn=100)
        ).data.T
        # only select key points' fpfh
        keyfeats1 = fpfhs1[keyptsindices1]
        keyfeats2 = fpfhs2[keyptsindices2]

        # step4: coarse ransac registration
        # use fpfh feature descriptor to compute matches
        matches = ransac.init_matches(keyfeats1.T, keyfeats2.T)
        correct = utils.ground_truth_matches(matches, keypts1, keypts2, args.voxel_size * 2.0, T_gdth) # 上帝视角
        correct_valid_num = correct.astype(np.int32).sum()
        correct_total_num = correct.shape[0]
        tqdm.write(utils.log_info(f"gdth/init: {correct_valid_num:.2f}/{correct_total_num:.2f}={correct_valid_num/correct_total_num:.2f}", quiet=True))
        
        matches_mat = np.concatenate([keyfeats1[matches[:, 0]], keyfeats2[matches[:, 1]]], axis=1)
        
        np.savez(f"{args.out_root}/{sample_name}", features=matches_mat, labels=correct)
