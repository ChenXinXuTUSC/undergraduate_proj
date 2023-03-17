import os
import numpy as np
import collections
import open3d as o3d

from datasets import datasets
import config
import utils
import models

if __name__ == "__main__":
    args = vars(config.args)

    available_datasets = {attr_name: getattr(datasets, attr_name) for attr_name in dir(datasets) if callable(getattr(datasets, attr_name))}
    dataloader = available_datasets[args["data_type"]](
        root=args["data_root"],
        shuffle=True,
        augment=True,
        augdgre=30.0,
        augdist=2.0
    )

    for points1, points2, T_gdth, sample_name in dataloader:
        utils.log_info(sample_name)
        # step1: voxel down sample
        points1 = utils.voxel_down_sample(points1, args["ICP_radius"] * 2.0)
        points2 = utils.voxel_down_sample(points2, args["ICP_radius"] * 2.0)

        # step2: detect key points using ISS
        keypoints1 = utils.iss_detect(points1, args["ICP_radius"])
        keypoints2 = utils.iss_detect(points2, args["ICP_radius"])
        if len(keypoints1["id"].values) == 0 or len(keypoints2["id"].values) == 0:
            utils.log_warn(f"{sample_name} failed to find ISS keypoints, continue to next sample")
            continue

        # step3: compute FCGF feature using pretrained model
