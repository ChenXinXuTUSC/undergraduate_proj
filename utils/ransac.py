import numpy as np
import open3d as o3d

import concurrent.futures
from scipy.spatial.distance import pdist

from . import tools
from .tools import *

def init_matches(srcfeatpts:np.ndarray, dstfeatpts:np.ndarray):
    '''
    the function assumes that points2 is target points
    '''
    fpfh_search_tree = o3d.geometry.KDTreeFlann(npy2o3d(dstfeatpts))

    rough_matches = []
    for src_idx, featpoint in srcfeatpts:
        _, dst_idx, _ = fpfh_search_tree.search_knn_vector_3d(featpoint[:3], 1)
        rough_matches.append([src_idx, dst_idx[0]])
    return np.asarray(rough_matches)

def one_iter_match(
        srcpts, dstpts, 
        proposal,
        checkr_params
    ):
    srcfeatidx = proposal[:0]
    dstfeatidx = proposal[:1]

    if checkr_params.normal_angle_threshold is not None:
        # we assume the point cloud feature is [x,y,z,r,g,b,u,v,w]
        src_normals = srcpts[srcfeatidx, 6:9]
        dst_normals = dstpts[dstfeatidx, 6:9]

        # 法向量约定俗成是单位向量，单位向量点积得到它们的夹角
        # np.dot是矩阵乘法，*是对应元素乘法
        match_cosine = (src_normals * dst_normals).sum(axis=1)
        # 如果每对匹配特征点的法向量夹角都符合阈值，则匹配通过
        is_valid_normal_match = np.all(match_cosine, checkr_params.normal_angle_threshold)
        if not is_valid_normal_match:
            return None
    
    srcfeatpts = srcpts[srcfeatidx]
    dstfeatpts = dstpts[dstfeatidx]
    # 使用pdist计算点集中两两点之间的欧氏距离
    src_mnn_dist = pdist(srcfeatpts[:3])
    dst_mnn_dist = pdist(dstfeatpts[:3])
    is_valid_mnn_match = np.all(
        np.logical_and(
            src_mnn_dist > dst_mnn_dist * checkr_params.max_mnn_dist_ratio,
            dst_mnn_dist > src_mnn_dist * checkr_params.max_mnn_dist_ratio
        )
    )
    if not is_valid_mnn_match:
        return None
    
    T = solve_procrustes(srcfeatpts, dstfeatpts)
    R, t = T[0:3, 0:3], T[0:3, 3:]

    # deviation to judge outlier and inlier
    deviation = np.linalg.norm(dstfeatpts - np.dot(srcfeatpts, R) - t, axis=1)
    is_valid_deviation_match = np.all(deviation <= checkr_params.max_deviation_norm)
    if not is_valid_deviation_match:
        return None
    
    return T

def ransac_match(
    srcpts: np.ndarray,
    dstpts: np.ndarray,
    srcfeatpts: np.ndarray,
    dstfeatpts: np.ndarray,
    ransac_params,
    checkr_params
):
    matches = init_matches(srcfeatpts, dstfeatpts)

    # build search tree on dst feature points
    dst_search_tree = o3d.geometry.KDTreeFlann(dstpts)

    initial_T = None
    # infinite generator
    proposal_generator = (matches[np.random.choice(range(matches.shape[0], ransac_params.num_samples, replace=False))] for _ in iter(int, 1))
    validator = lambda proposal: one_iter_match(srcpts, dstpts, proposal, checkr_params)

    # concurrently find the inital T, using 'with' method
    # so that Executor.shutdown(wait=True) is not needed.
    with concurrent.futures.ThreadPoolExecutor(max_workers=ransac_params.max_workers) as executor:
        for T in executor.map(validator, proposal_generator):
            if T is not None:
                log_info(f"find initial transformation:{T}")
                break
    
    # baseline
    
    for i in range(ransac_params.max_iteration):
        T = validator(next(proposal_generator))

