import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)) , ".."))

import numpy as np
import torch
import MinkowskiEngine as ME

from tqdm import tqdm
from easydict import EasyDict as edict

from datasets import datasets
import config
import utils
from utils import ransac
import models


if __name__ == "__main__":
    args = config.args

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

    model_conf = torch.load(args.extracter_weight)["config"]
    model_params = torch.load(args.extracter_weight)["state_dict"]
    feat_model = models.fcgf.load_model(args.fcgf_model)(
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
    for i, (points1, points2, T_gdth, sample_name) in tqdm(enumerate(dataloader), total=len(dataloader), ncols=100, desc="gen npz"):
        # step1: voxel downsample
        downsampled_coords1, voxelized_coords1, idx_dse2vox1 = utils.voxel_down_sample_gpt(points1, args.voxel_size)
        downsampled_coords2, voxelized_coords2, idx_dse2vox2 = utils.voxel_down_sample_gpt(points2, args.voxel_size)

        # step2: detect key points using ISS
        keyptsdict1 = utils.iss_detect(downsampled_coords1, args.voxel_size * args.key_radius_factor, args.lambda1, args.lambda2)
        keyptsdict2 = utils.iss_detect(downsampled_coords2, args.voxel_size * args.key_radius_factor, args.lambda1, args.lambda2)
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
        keyfcgfs1 = fcgfs1[keyptsindices1]
        keyfcgfs2 = fcgfs2[keyptsindices2]

        # step4: coarse ransac registration
        # use fpfh feature descriptor to compute matches
        matches = ransac.init_matches(keyfcgfs1.T, keyfcgfs2.T)
        correct = utils.ground_truth_matches(matches, keypts1, keypts2, args.voxel_size * 1.50, T_gdth)
        num_valid_matches = correct.astype(np.int32).sum()
        num_total_matches = correct.shape[0]
        tqdm.write(utils.log_info(f"gdth/init: {num_valid_matches:.2f}/{num_total_matches:.2f}={num_valid_matches/num_total_matches:.2f}", quiet=True))
        
        matches_mat = np.concatenate([keyfcgfs1[matches[:, 0]], keyfcgfs2[matches[:, 1]]], axis=1)
        
        np.savez(f"{args.out_root}/{sample_name}", features=matches_mat, labels=correct)
