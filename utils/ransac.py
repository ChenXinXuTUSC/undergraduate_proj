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

def filter_matches(matches:np.ndarray, keyfpfhs1:np.ndarray, keyfpfhs2:np.ndarray):
    avg_fdist = np.sqrt(((keyfpfhs1[:, matches[:, 0]] - keyfpfhs2[:, matches[:, 1]]) ** 2).sum(axis=1)) / len(matches)
    avg_fdist = np.reshape(avg_fdist, (len(avg_fdist), 1))
    all_fdist = np.sqrt(((keyfpfhs1[:, matches[:, 0]] - keyfpfhs2[:, matches[:, 1]]) ** 2))
    valid_mask = (all_fdist < avg_fdist).astype(np.int32).sum(axis=0) > 20
    return matches[valid_mask]

def one_iter_match(
        srckts, dstkts, 
        proposal,
        checkr_params
    ):
    # 把两组关键点按对应关系排列好
    srckts = srckts[proposal[:,0]]
    dstkts = dstkts[proposal[:,1]]
    if checkr_params.normal_angle_threshold is not None:
        # we assume the point cloud feature is [x,y,z,r,g,b,u,v,w]
        src_normals = srckts[6:9]
        dst_normals = dstkts[6:9]

        # 法向量约定俗成是单位向量，单位向量点积得到它们的夹角
        # np.dot是矩阵乘法，*是对应元素乘法
        match_cosine = (src_normals * dst_normals).sum(axis=1)
        # 如果每对匹配特征点的法向量夹角都符合阈值，则匹配通过
        is_valid_normal_match = np.all(match_cosine, checkr_params.normal_angle_threshold)
        if not is_valid_normal_match:
            return None
    

    srckts = srckts[:,:3]
    dstkts = dstkts[:,:3]
    # 使用pdist计算点集中两两点之间的欧氏距离，这是一个强假设
    # 对于很多匹配情况是无法通过的，比如一旦原点集有两个关键点
    # 匹配到了目标点集的同一个点，那么这个假设检测就无法通过。
    src_mnn_dist = pdist(srckts) # 只取用xyz坐标部分
    dst_mnn_dist = pdist(dstkts) # 只取用xyz坐标部分
    is_valid_mnn_match = np.all(
        np.logical_and(
            src_mnn_dist > dst_mnn_dist * checkr_params.max_mnn_dist_ratio,
            dst_mnn_dist > src_mnn_dist * checkr_params.max_mnn_dist_ratio
        )
    )
    if not is_valid_mnn_match:
        return None
    
    T = solve_procrustes(srckts, dstkts)
    R, t = T[0:3, 0:3], T[0:3, 3]

    # deviation to judge outlier and inlier
    deviation = np.linalg.norm(dstkts - np.dot(srckts, R) - t, axis=1)
    is_valid_deviation_match = np.all(deviation <= checkr_params.max_correspondence_dist)
    if not is_valid_deviation_match:
        return None
    
    return T

def ransac_match(
        srckts: np.ndarray,
        dstkts: np.ndarray,
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
    dst_search_tree = o3d.geometry.KDTreeFlann(npy2o3d(dstkts))

    initial_T = None
    # infinite generator
    proposal_generator = (matches[np.random.choice(range(matches.shape[0]), ransac_params.num_samples, replace=False)] for _ in iter(int, 1))
    validator = lambda proposal: one_iter_match(srckts, dstkts, proposal, checkr_params)

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
        srckts, dstkts, dst_search_tree,
        initial_T,
        ransac_params.max_correspondence_dist, ransac_params.max_refine_num
    )
    num_validation = 0
    for _ in tqdm(range(ransac_params.max_iter_num), total=ransac_params.max_iter_num, ncols=100, desc="ICP refining"):
        T = validator(next(proposal_generator))
        # check validity
        curr_res = icp.ICP_exact_match(
            srckts, dstkts, dst_search_tree,
            T,
            ransac_params.max_corresponding_dist, ransac_params.max_refine_num
        )
        if curr_res.fitness < best_res.fitness:
            best_res = curr_res
        # if T is not None and num_validation < ransac_params.max_valid_num:
        #     num_validation += 1
        
        # if num_validation == ransac_params.max_valid_num:
        #     break
    log_info(f"ICP refinement: {best_res}")
    return best_res


def one_iter_match_copy(
    source_idx, target_idx,
    source_pcd, target_pcd,
    proposal,
    checkr_params
):
    source_idx, target_idx = proposal[:, 0], proposal[:, 1]
    #法向量校准
    if not checkr_params.normal_angle_threshold is None:
        # get corresponding normals:
        normals_source = np.asarray(source_pcd.normals)[source_idx]
        normals_target = np.asarray(target_pcd.normals)[target_idx]

        # a. normal direction check:
        normal_cos_distances = (normals_source*normals_target).sum(axis = 1)
        is_valid_normal_match = np.all(normal_cos_distances >= np.cos(checkr_params.normal_angle_threshold))

        if not is_valid_normal_match:
            return None

    # 按对应关系排列好关键点对
    points_source = np.asarray(source_pcd.points)[source_idx]
    points_target = np.asarray(target_pcd.points)[target_idx]

    # 关键点群的特征通过同一点群内的互相距离来表示，只有选择的
    # 两个关键点群特征相似时，认为是好的proposal
    # 构建距离矩阵，使用 Mutual nearest descriptor matching
    pdist_source = pdist(points_source)
    pdist_target = pdist(points_target)
    pdist_source_valid_mean = np.mean((pdist_source > checkr_params.max_mnn_dist_ratio * pdist_target).astype(float))
    pdist_target_valid_mean = np.mean((pdist_target > checkr_params.max_mnn_dist_ratio * pdist_source).astype(float))
    is_valid_mnn_dist = pdist_source_valid_mean > checkr_params.max_mnn_dist_ratio and pdist_target_valid_mean > checkr_params.max_mnn_dist_ratio

    if not is_valid_mnn_dist:
        return None

    # fast correspondence distance check
    T = solve_procrustes(points_source, points_target) # 通过 svd 初步求解 旋转、平移矩阵
    R, t = T[:3, :3], T[:3, 3]
    # 通过距离偏差判断 inliers outliers
    deviation = np.linalg.norm(
        points_target - np.dot(points_source, R.T) - t,
        axis = 1
    )
    #判断数目
    is_valid_correspondence_distance = np.all(deviation <= checkr_params.max_correspondence_dist)

    return T if is_valid_correspondence_distance else None

def ransac_match_copy(
        source_idx, target_idx,
        source_pcd, target_pcd,
        source_fds, target_fds,
        ransac_params, checkr_params
    ):
    # step5.1 Establish correspondences(point pairs) 建立 pairs
    matches = init_matches(source_fds, target_fds) # 通过 fpfh 建立的feature squre map 建立最初的 pairs

    # build search tree on the target:
    search_tree_target = o3d.geometry.KDTreeFlann(target_pcd)
    # FLANN stands for fast library for aproximate nearest neighbours

    T = None
    # step 5.2 select 4 pairs at each iteration,选择4对corresponding 进行模型拟合
    proposal_generator = (
        matches[np.random.choice(range(matches.shape[0]), ransac_params.num_samples, replace=False)] for _ in iter(int, 1)
    )
    # step 5.3 iter 迭代，iter_match() ,选择出 vaild T
    validator = lambda proposal: one_iter_match_copy(source_idx, target_idx, source_pcd, target_pcd, proposal, checkr_params)
    for T in map(validator, proposal_generator):
        if T is not None:
            log_dbug(f"initial transformation found:\n{T}")
            tmp_R = T[:3, :3]
            log_dbug(f"validate if R is  orthogonal:\n{np.dot(tmp_R, tmp_R.T)}")
            break

    #set baseline
    log_dbug('finding first valid proposal...')
    best_result = icp.ICP_exact_match_copy(
        source_pcd, target_pcd, search_tree_target,
        T,
        ransac_params.max_correspondence_dist,
        ransac_params.max_refine_num
    )

    # RANSAC:
    num_validation = 0
    for _ in tqdm(range(ransac_params.max_iter_num), total=ransac_params.max_iter_num, ncols=100, desc="ransac finding"):
        # get proposal:
        T = validator(next(proposal_generator))

        # check validity:
        if T is not None and num_validation < ransac_params.max_valid_num:
            num_validation += 1

            # refine estimation on all keypoints:
            result = icp.ICP_exact_match_copy(
                source_pcd, target_pcd, search_tree_target,
                T,
                ransac_params.max_correspondence_dist,
                ransac_params.max_refine_num
            )

            # update best result:
            if best_result.fitness < result.fitness:
                best_result = result

            if num_validation == ransac_params.max_valid_num:
                break

    return best_result
