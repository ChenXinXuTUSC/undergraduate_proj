import numpy as np
import open3d as o3d
from easydict import EasyDict as edict

from datasets import datasets
import config
import utils

from utils import ransac
import models

import MinkowskiEngine as ME
import torch

import matplotlib.pyplot as plt

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
        utils.log_info(sample_name)
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
        keyfcgfs1 = fcgfs1[keyptsdict1["id"].values]
        keyfcgfs2 = fcgfs2[keyptsdict2["id"].values]

        # use fpfh feature descriptor to compute matches
        matches = ransac.init_matches(keyfcgfs1.T, keyfcgfs2.T)
        correct = utils.ground_truth_matches(matches, keypts1, keypts2, args.ICP_radius * 2.5, T_gdth) # 上帝视角
        correct_valid_num = correct.astype(np.int32).sum()
        correct_total_num = correct.shape[0]
        utils.log_info(f"gdth/init: {correct_valid_num:.2f}/{correct_total_num:.2f}={correct_valid_num/correct_total_num:.2f}")

        # 将对匹配对索引从关键点集合映射回原点云集合
        init_matches = np.array([keyptsdict1["id"].values[matches[:,0]], keyptsdict2["id"].values[matches[:,1]]]).T
        gdth_matches = np.array([keyptsdict1["id"].values[matches[:,0]], keyptsdict2["id"].values[matches[:,1]]]).T[correct]

        # principle component analysis
        keyfcgfs1 = np.matmul(keyfcgfs1, utils.principle_K_components(keyfcgfs1, 3))
        keyfcgfs2 = np.matmul(keyfcgfs2, utils.principle_K_components(keyfcgfs2, 3))

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(keyfcgfs1[:,0],keyfcgfs1[:,1],keyfcgfs1[:,2], color=(1.0, 1.0, 0.0), marker='o', label="points1", alpha=0.2)
        ax.scatter(keyfcgfs2[:,0],keyfcgfs2[:,1],keyfcgfs2[:,2], color=(0.0, 1.0, 1.0), marker='^', label="points2", alpha=0.2)
        ax.legend(loc="best")
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.savefig(f"{args.out_root}/fcgf_{args.data_type}_visual_3d.jpg")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(keyfcgfs1[:,0],keyfcgfs1[:,1], color=(1.0, 0.5, 0.0), marker='o', label="points1", alpha=0.2)
        ax.scatter(keyfcgfs2[:,0],keyfcgfs2[:,1], color=(0.0, 0.5, 1.0), marker='^', label="points2", alpha=0.2)
        ax.legend(loc="best")
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        plt.savefig(f"{args.out_root}/fcgf_{args.data_type}_visual_2d.jpg")

        break
