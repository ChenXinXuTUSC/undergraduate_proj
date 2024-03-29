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

# step1: read point cloud pair
# step2: voxel down sample
# step3: extract ISS feature
# step4: feature description
# step5: RANSAC registration
#   step5.1: establish feature correspondences
#   step5.2: select n(> 3) pairs to solve the transformation R and t
#   step5.3: repeat step5.2 until error converge
# step6: ICP optimized transformation [R,t]

#RANSAC configuration:
RANSACCONF = collections.namedtuple(
    "RANSACCONF",
    [
        "max_workers",
        "num_samples",
        "max_corresponding_dist", 'max_iter_num', 'max_valid_num', 'max_refine_num'
    ]
)
# fast pruning algorithm configuration:
CHECKRCONF = collections.namedtuple(
    "CHECKRCONF",
    [
        "max_corresponding_dist",
        "max_mnn_dist_ratio", 
        "normal_angle_threshold"
    ]
)

model_classes = [
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "monitor",
    "night_stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox",
]
statistics = {
    mclsn:{
        "totl_cnt":0,
        "init_set":[],
        "gdth_set":[],
        "corr_set":[],
        "err_R":[],
        "err_t":[],
        "err_x":[]
    } for mclsn in model_classes
}


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

    sample_num = 100
    sample_cnt = 0
    for points1, points2, T_gdth, sample_name in dataloader:
        sample_cnt += 1
        if sample_cnt == sample_num:
            utils.log_info("finish sampling")
            break
        
        utils.log_info(f"processing {sample_name}")
        mclsn = "_".join(sample_name.split("_")[:-1])
        statistics[mclsn]["totl_cnt"] += 1
        # modelnet40 dataset doesn't contain normals
        points1_o3d = utils.npy2o3d(points1)
        points1_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.ICP_radius*2.0, max_nn=30))
        points1 = utils.o3d2npy(points1_o3d)
        points2_o3d = utils.npy2o3d(points2)
        points2_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.ICP_radius*2.0, max_nn=30))
        points2 = utils.o3d2npy(points2_o3d)
        
        
        # step1: voxel downsample
        downsampled_coords1, voxelized_coords1, idx_dse2vox1 = utils.voxel_down_sample_gpt(points1, args.ICP_radius)
        downsampled_coords2, voxelized_coords2, idx_dse2vox2 = utils.voxel_down_sample_gpt(points2, args.ICP_radius)

        # step2: detect key points using ISS
        keyptsdict1 = utils.iss_detect(downsampled_coords1, args.ICP_radius * 2.0)
        keyptsdict2 = utils.iss_detect(downsampled_coords2, args.ICP_radius * 2.0)
        if len(keyptsdict1["id"].values) == 0 or len(keyptsdict2["id"].values) == 0:
            utils.log_warn(f"{sample_name} failed to find ISS keypoints, continue to next sample")
            continue
        keypts1 = downsampled_coords1[keyptsdict1["id"].values]
        keypts2 = downsampled_coords2[keyptsdict2["id"].values]

        # step3: compute FPFH for all points
        fpfhs1 = o3d.pipelines.registration.compute_fpfh_feature(
            utils.npy2o3d(downsampled_coords1),
            o3d.geometry.KDTreeSearchParamHybrid(radius=args.ICP_radius*1.5, max_nn=30)
        ).data
        fpfhs2 = o3d.pipelines.registration.compute_fpfh_feature(
            utils.npy2o3d(downsampled_coords2),
            o3d.geometry.KDTreeSearchParamHybrid(radius=args.ICP_radius*1.5, max_nn=30)
        ).data
        # select key points' fpfh feature
        keyfeats1 = fpfhs1[:, keyptsdict1["id"].values]
        keyfeats2 = fpfhs2[:, keyptsdict2["id"].values]

        matches = ransac.init_matches(keyfeats1, keyfeats2)
        correct = utils.ground_truth_matches(matches, keypts1, keypts2, args.ICP_radius * 1.5, T_gdth) # 上帝视角
        correct_valid_num = correct.astype(np.int32).sum().item()
        correct_total_num = correct.shape[0]
        utils.log_info(f"gdth/init: {correct_valid_num:.2f}/{correct_total_num:.2f}={correct_valid_num/correct_total_num:.2f}")
        # 将对匹配对索引从关键点集合映射回原点云集合
        init_matches = np.array([keyptsdict1["id"].values[matches[:,0]], keyptsdict2["id"].values[matches[:,1]]]).T
        gdth_matches = np.array([keyptsdict1["id"].values[matches[:,0]], keyptsdict2["id"].values[matches[:,1]]]).T[correct]
        
        if int(correct_total_num) < 4 or int(correct_valid_num) < 4:
            utils.log_warn(f"{sample_name} failed to find enough matches")
            continue
        
        statistics[mclsn]["init_set"].append(correct_total_num)
        statistics[mclsn]["gdth_set"].append(correct_valid_num)

        # step4: ransac initial registration
        initial_ransac = utils.ransac_match(
            keypts1, keypts2,
            keyfeats1, keyfeats2,
            ransac_params=RANSACCONF(
                max_workers=4, num_samples=4,
                max_corresponding_dist=args.ICP_radius*2.0,
                max_iter_num=2000, max_valid_num=100, max_refine_num=30
            ),
            checkr_params=CHECKRCONF(
                max_corresponding_dist=args.ICP_radius*2.0,
                max_mnn_dist_ratio=0.80,
                normal_angle_threshold=None
            ),
            matches=matches
        )

        if len(initial_ransac.correspondence_set) == 0:
            utils.log_warn(sample_name, "failed to recover the transformation")
            continue
        statistics[mclsn]["corr_set"].append(len(initial_ransac.correspondence_set))
        final_result = icp.ICP_exact_match(
            downsampled_coords1, downsampled_coords2,
            o3d.geometry.KDTreeFlann(utils.npy2o3d(downsampled_coords2)), 
            initial_ransac.transformation, args.ICP_radius,
            10
        )
        T_pred = final_result.transformation
        
        axis_pred, degr_pred = utils.resolve_axis_angle(T_pred, deg=True)
        axis_gdth, degr_gdth = utils.resolve_axis_angle(T_gdth, deg=True)
        statistics[mclsn]["err_R"].append(np.absolute(degr_gdth - degr_pred))
        statistics[mclsn]["err_t"].append((T_gdth[:3,3] - T_pred[:3,3]).sum())
        statistics[mclsn]["err_x"].append(np.arccos(np.dot(axis_gdth, axis_pred)) * 180.0 / np.pi)
        utils.log_info(f"pred: axis:{axis_pred}, rot_deg:{degr_pred}, trans:{T_pred[:3,3]}")
        utils.log_info(f"gdth: axis:{axis_gdth}, rot_deg:{degr_gdth}, trans:{T_gdth[:3,3]}")


        # do some visualization works
        utils.dump_registration_result(
            args.out_root, sample_name,
            points1, points2,
            downsampled_coords1, keyptsdict1["id"].values,
            downsampled_coords2, keyptsdict2["id"].values,
            T_gdth, T_pred, gdth_matches
        )
        utils.log_info(f"finish processing {sample_name}")
        break # only for test

# import json
# metrics = json.dumps(statistics, sort_keys=True, indent=4, separators=(',', ': '))
# with open("./fpfh_modelnet40_metrics.json", 'w') as f:
#     f.write(metrics)
