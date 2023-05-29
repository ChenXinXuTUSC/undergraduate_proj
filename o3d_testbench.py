import os

import open3d as o3d
import numpy as np

import time

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
    distance_threshold,
    ransac_n=3
    ):
    '''
    http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
    '''
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.85),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def execute_match_registration(
    source_down, target_down,
    matches,
    distance_threshold,
    ransac_n=3
):
    if type(matches) == np.ndarray:
        matches = o3d.utility.Vector2iVector(matches)
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source_down, target_down,
        matches, distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 
        ransac_n=ransac_n,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 80000)
    )
    
    return result

if __name__ == "__main__":
    args = edict(vars(config.args))

    available_datasets = {attr_name: getattr(datasets, attr_name) for attr_name in dir(datasets) if callable(getattr(datasets, attr_name))}
    dataloader = available_datasets[args.data_type](
        root=args.data_root,
        shuffle=False,
        augdict= edict({
            "augment": True,
            "augdgre": 60.00,
            "augdist": 5.00,
            "augjitr": 0.00,
            "augnois": 0.00
        }),
        args=args
    )
    
    # timestamp = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
    # dumpfile = open(os.path.join(args.out_root, f"{args.data_type}_count_{timestamp}.txt"), 'w')
    
    total = len(dataloader)
    for i, (points1, points2, T_gdth, sample_name) in enumerate(dataloader):
        utils.log_info(f"{i + 1:4d}/{total} {sample_name}")
        if args.recompute_norm:
            points1_o3d = utils.npy2o3d(points1)
            points1_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2.0, max_nn=100))
            points1 = utils.o3d2npy(points1_o3d)
            points2_o3d = utils.npy2o3d(points2)
            points2_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2.0, max_nn=100))
            points2 = utils.o3d2npy(points2_o3d)
        
        downsampled_coords1, voxelized_coords1, idx_dse2vox1 = utils.voxel_down_sample_gpt(points1, args.voxel_size)
        downsampled_coords2, voxelized_coords2, idx_dse2vox2 = utils.voxel_down_sample_gpt(points2, args.voxel_size)
        downsampled_coords1_o3d = utils.npy2o3d(downsampled_coords1)
        downsampled_coords2_o3d = utils.npy2o3d(downsampled_coords2)
        fpfhs1 = o3d.pipelines.registration.compute_fpfh_feature(
            downsampled_coords1_o3d,
            o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * args.fpfh_radius_factor, max_nn=args.fpfh_nn)
        ).data
        fpfhs2 = o3d.pipelines.registration.compute_fpfh_feature(
            downsampled_coords2_o3d,
            o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * args.fpfh_radius_factor, max_nn=args.fpfh_nn)
        ).data
        
        matches = utils.init_matches(fpfhs1, fpfhs2)
        t1 = time.time()
        result = execute_match_registration(
            downsampled_coords1_o3d, downsampled_coords2_o3d,
            matches, args.voxel_size * 1.5,
            ransac_n=3
        )
        t2 = time.time()
        # result = execute_global_registration(
        #     downsampled_coords1_o3d, downsampled_coords2_o3d,
        #     fpfhs1, fpfhs2,
        #     args.voxel_size * 1.5
        # )
        
        T_pred = result.transformation
        
        diff_raxis, diff_rdegr, diff_trans = utils.trans_diff(T_pred, T_gdth)
        
        # # raxis rdegr
        # dumpfile.write(f"{sample_name} {diff_raxis:5.3f} {diff_rdegr:5.3f} ")
        # # trans
        # for x in (diff_trans):
        #     dumpfile.write(f"{x:5.3f} ")
        # # times
        # dumpfile.write(f"{t2 - t1:5.3f}")
        # dumpfile.write('\n')
        # dumpfile.flush()
        
        # if you are going to save the fuse result of two point sets
        # do the translation on point2 instead of point1, it's open-
        # 3d's feature
        points1[:, 3:6] = np.array([255, 255, 0])
        points2[:, 3:6] = np.array([0, 255, 255])
        
        points2[:, :3] = points2[:, :3] - T_pred[:3,  3]
        points2[:, :3] = points2[:, :3] @ T_pred[:3, :3]
        utils.fuse2frags(
            points1, points2,
            utils.make_ply_vtx_type(True, True),
            args.out_root, "o3dpredfuse.ply"
        )
    # dumpfile.close()
