import open3d as o3d
import numpy as np

import utils
from easydict import EasyDict as edict

import config
from datasets import datasets

# step1: read point cloud pair
# step2: voxel down sample
# step3: extract ISS feature
# step4: feature description
# step5: RANSAC registration
#   step5.1: establish feature correspondences
#   step5.2: select n(> 3) pairs to solve the transformation R and t
#   step5.3: repeat step5.2 until error converge
# step6: ICP optimized transformation [R,t]

def execute_global_registration(
    source_down, target_down, 
    source_fpfh, target_fpfh,
    voxel_size
    ):
    '''
    http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
    '''
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

if __name__ == "__main__":
    args = edict(vars(config.args))

    available_datasets = {attr_name: getattr(datasets, attr_name) for attr_name in dir(datasets) if callable(getattr(datasets, attr_name))}
    dataloader = available_datasets[args.data_type](
        root=args.data_root,
        shuffle=True,
        augdict= edict({
            "augment": True,
            "augdgre": 5.00,
            "augdist": 5.0,
            "augjitr": 0.00,
            "augnois": 0
        }),
        args=args
    )
    
    for points1, points2, T_gdth, sample_name in dataloader:
        utils.log_info(sample_name)
        
        points1_o3d = utils.npy2o3d(points1)
        points1_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2.0, max_nn=30))
        points1 = utils.o3d2npy(points1_o3d)
        points2_o3d = utils.npy2o3d(points2)
        points2_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2.0, max_nn=30))
        points2 = utils.o3d2npy(points1_o3d)
        
        downsampled_coords1, voxelized_coords1, idx_dse2vox1 = utils.voxel_down_sample_gpt(points1, args.voxel_size)
        downsampled_coords2, voxelized_coords2, idx_dse2vox2 = utils.voxel_down_sample_gpt(points2, args.voxel_size)
        downsampled_coords1_o3d = utils.npy2o3d(downsampled_coords1)
        downsampled_coords2_o3d = utils.npy2o3d(downsampled_coords2)
        fpfhs1 = o3d.pipelines.registration.compute_fpfh_feature(
            downsampled_coords1_o3d,
            o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * args.fpfh_radius_factor, max_nn=args.fpfh_nn)
        )
        fpfhs2 = o3d.pipelines.registration.compute_fpfh_feature(
            downsampled_coords2_o3d,
            o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * args.fpfh_radius_factor, max_nn=args.fpfh_nn)
        )
        
        result = execute_global_registration(
            downsampled_coords1_o3d, downsampled_coords2_o3d,
            fpfhs1, fpfhs2,
            args.voxel_size
        )
        
        T_pred = result.transformation
        
        raxis_pred, rdegr_pred = utils.resolve_axis_angle(T_pred, deg=True)
        raxis_gdth, rdegr_gdth = utils.resolve_axis_angle(T_gdth, deg=True)
        trans_pred, trans_gdth = T_pred[:3, 3], T_gdth[:3, 3]
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
