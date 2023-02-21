import numpy as np
import open3d as o3d

import concurrent.futures
from scipy.spatial.distance import pdist

from tqdm import tqdm
import time

from . import tools
from .tools import *
from . import icp

def init_matches(srcfds:np.ndarray, dstfds:np.ndarray):
    '''
    the function assumes that points2 is target points
    '''
    fpfh_search_tree = o3d.geometry.KDTreeFlann(dstfds)
    num_srcfds = srcfds.shape[1]
    rough_matches = []
    for col_idx in range(num_srcfds):
        query = srcfds[:, col_idx]
        # KD树搜索的查询目标，必须具有和构建树时相同的维度K，如果对KD树查询
        # 时指定了向量的长度，那么只需要输入向量的元素列表而无需关心是列向量
        # 还是行向量，但如果是使用x-dimension的话，似乎必须使用列向量？但调
        # 试过后发现好像也不用列向量啊？？？
        _, dst_idx, _ = fpfh_search_tree.search_knn_vector_xd(query, 1)
        rough_matches.append([col_idx, dst_idx[0]])
    return np.asarray(rough_matches)

def one_iter_match(
        srcpts, dstpts, 
        proposal,
        checkr_params
    ):
    srcfdsidx = proposal[:, 0]
    dstfdsidx = proposal[:, 1]

    if checkr_params.normal_angle_threshold is not None:
        # we assume the point cloud feature is [x,y,z,r,g,b,u,v,w]
        src_normals = srcpts[srcfdsidx, 6:9]
        dst_normals = dstpts[dstfdsidx, 6:9]

        # 法向量约定俗成是单位向量，单位向量点积得到它们的夹角
        # np.dot是矩阵乘法，*是对应元素乘法
        match_cosine = (src_normals * dst_normals).sum(axis=1)
        # 如果每对匹配特征点的法向量夹角都符合阈值，则匹配通过
        is_valid_normal_match = np.all(match_cosine, checkr_params.normal_angle_threshold)
        if not is_valid_normal_match:
            return None
    
    srcfeatpts = srcpts[srcfdsidx, :3]
    dstfeatpts = dstpts[dstfdsidx, :3]
    # 使用pdist计算点集中两两点之间的欧氏距离
    src_mnn_dist = pdist(srcfeatpts)
    dst_mnn_dist = pdist(dstfeatpts)
    is_valid_mnn_match = np.all(
        np.logical_and(
            src_mnn_dist > dst_mnn_dist * checkr_params.max_mnn_dist_ratio,
            dst_mnn_dist > src_mnn_dist * checkr_params.max_mnn_dist_ratio
        )
    )
    if not is_valid_mnn_match:
        return None
    
    T = solve_procrustes(srcfeatpts, dstfeatpts)
    R, t = T[0:3, 0:3], T[0:3, 3]

    # deviation to judge outlier and inlier
    deviation = np.linalg.norm(dstfeatpts - np.dot(srcfeatpts, R) - t, axis=1)
    is_valid_deviation_match = np.all(deviation <= checkr_params.max_correspondence_dist)
    if not is_valid_deviation_match:
        return None
    
    return T

def ransac_match(
        srcpts: np.ndarray,
        dstpts: np.ndarray,
        srcfds: np.ndarray,
        dstfds: np.ndarray,
        ransac_params,
        checkr_params
    ):
    '''
    params
    -
    * srcpts - source point cloud, nx3 numpy ndarray
    * dstpts - target point cloud, nx3 numpy ndarray
    * srcfds - source feature descriptors, nx33 ndarray for FPFH descriptor
    * dstfds - target feature descriptors, nx33 ndarray for FPFH descriptor

    return
    -
    * registration_res - contains 4x4 SE(3) and nx2 corresponding feat set
    '''
    matches = init_matches(srcfds, dstfds)

    # build search tree on dst feature points
    dst_search_tree = o3d.geometry.KDTreeFlann(npy2o3d(dstpts))

    initial_T = None
    # infinite generator
    proposal_generator = (matches[np.random.choice(range(matches.shape[0]), ransac_params.num_samples, replace=False)] for _ in iter(int, 1))
    validator = lambda proposal: one_iter_match(srcpts, dstpts, proposal, checkr_params)

    # concurrently find the inital T, using 'with' method
    # so that Executor.shutdown(wait=True) is not needed.
    log_dbug("finding initial transformation T...")
    # 这里使用多线程似乎会出问题。。。会卡住不动
    # with concurrent.futures.ThreadPoolExecutor(max_workers=ransac_params.max_workers) as executor:
    t1 = time.time()
    for T in map(validator, proposal_generator):
        if T is not None:
            log_dbug(f"initial transformation found:\n{T}")
            tmp_R = T[:3, :3]
            log_dbug(f"validate if R is  orthogonal:\n{np.dot(tmp_R, tmp_R.T)}")
            break
    t2 = time.time()
    log_info(f"finding proper T costs {t2-t1:.2f}s")
    initial_T = T
    
    # baseline
    best_res = icp.ICP_exact_match(
        srcpts, dstpts, dst_search_tree,
        initial_T,
        ransac_params.max_correspondence_dist, ransac_params.max_refine_num
    )
    num_validation = 0
    for _ in tqdm(range(ransac_params.max_iter_num), total=ransac_params.max_iter_num, ncols=100, desc="ICP refining"):
        T = validator(next(proposal_generator))
        if T is not None and num_validation < ransac_params.max_validation_num:
            # check validity
            curr_res = icp.ICP_exact_match(
                srcpts, dstpts, dst_search_tree,
                T,
                ransac_params.max_corresponding_dist, ransac_params.max_refine_num
            )
            num_validation += 1
            if curr_res.fitness < best_res.fitness:
                best_res = curr_res
        
        if num_validation == ransac_params.max_valid_num:
            break
    log_info(f"ICP refinement: {best_res}")
    return best_res
