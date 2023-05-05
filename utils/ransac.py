import torch
import numpy as np
import open3d as o3d
import time
from tqdm import tqdm

from scipy.spatial.distance import pdist
import concurrent


from .tools import *
from . import icp

def init_matches(srcfds:np.ndarray, dstfds:np.ndarray):
    '''
    the function assumes that points2 is target points
    '''
    if type(srcfds) == torch.Tensor:
        srcfds = srcfds.detach().numpy()
    if type(dstfds) == torch.Tensor:
        dstfds = dstfds.detach().numpy()
    search_tree = o3d.geometry.KDTreeFlann(dstfds)
    num_srcfds = srcfds.shape[1]
    rough_matches = []
    for col_idx in range(num_srcfds):
        query = srcfds[:, col_idx]
        # KD树搜索的查询目标，必须具有和构建树时相同的维度K，如果对KD树查询
        # 时指定了向量的长度，那么只需要输入向量的元素列表而无需关心是列向量
        # 还是行向量，但如果是使用x-dimension的话，似乎必须使用列向量？但调
        # 试过后发现好像也不用列向量啊？？？
        _, dst_idx, _ = search_tree.search_knn_vector_xd(query, 1)
        rough_matches.append([col_idx, dst_idx[0]])
    
    return np.asarray(rough_matches)

def filter_matches(matches:np.ndarray, keyfpfhs1:np.ndarray, keyfpfhs2:np.ndarray):
    matches = copy.deepcopy(matches)
    avg_fdist = np.sqrt(((keyfpfhs1[:, matches[:, 0]] - keyfpfhs2[:, matches[:, 1]]) ** 2).sum(axis=1)) / len(matches)
    avg_fdist = np.reshape(avg_fdist, (len(avg_fdist), 1))
    all_fdist = np.sqrt(((keyfpfhs1[:, matches[:, 0]] - keyfpfhs2[:, matches[:, 1]]) ** 2))
    valid_mask = (all_fdist < avg_fdist).astype(np.int32).sum(axis=0) > 10
    return matches[valid_mask]

def one_iter_match(
        srckts, dstkts, 
        proposal,
        checkr_conf
    ):
    # 把两组关键点按对应关系排列好
    srckts = srckts[proposal[:,0]]
    dstkts = dstkts[proposal[:,1]]
    if checkr_conf.normdegr_thresh is not None:
        # we assume the point cloud feature is [x,y,z,r,g,b,u,v,w]
        src_normals = srckts[6:9]
        dst_normals = dstkts[6:9]

        # 法向量约定俗成是单位向量，单位向量点积得到它们的夹角
        # np.dot是矩阵乘法，*是对应元素乘法
        match_cosine = (src_normals * dst_normals).sum(axis=1)
        # 如果每对匹配特征点的法向量夹角都符合阈值，则匹配通过
        is_valid_normal_match = np.all(match_cosine >= checkr_conf.normdegr_thresh)
        if not is_valid_normal_match:
            return None
    
    srckts = srckts[:,:3]
    dstkts = dstkts[:,:3]
    # 计算关键点之间的相互间隔
    src_mnn_dist = pdist(srckts) # 只取用xyz坐标部分
    dst_mnn_dist = pdist(dstkts) # 只取用xyz坐标部分
    
    # 所选择的关键点应该要足够散开，否则聚集在一起的关键点容易
    # 产生局部最优解
    is_scatter_enough = np.logical_and(
        np.all(src_mnn_dist > checkr_conf.max_corrdist * 1.5),
        np.all(dst_mnn_dist > checkr_conf.max_corrdist * 1.5),
    )
    if not is_scatter_enough:
        return None
    # 使用pdist计算点集中两两点之间的欧氏距离，这是一个强假设
    # 对于很多匹配情况是无法通过的，比如一旦原点集有两个关键点
    # 匹配到了目标点集的同一个点，那么这个假设检测就无法通过。
    is_valid_mnn_match = np.all(
        np.logical_and(
            src_mnn_dist > dst_mnn_dist * checkr_conf.mutldist_factor,
            dst_mnn_dist > src_mnn_dist * checkr_conf.mutldist_factor
        )
    )
    if not is_valid_mnn_match:
        return None
    
    T = solve_procrustes(srckts, dstkts)
    R = T[:3,:3]
    t = T[:3, 3]

    # deviation to judge outlier and inlier
    deviation = np.linalg.norm(dstkts - np.dot(srckts, R) - t, axis=1)
    is_valid_deviation_match = np.all(deviation <= checkr_conf.max_corrdist)
    if not is_valid_deviation_match:
        return None
    
    return T

def ransac_match(
        srckts: np.ndarray,
        dstkts: np.ndarray,
        srcfds: np.ndarray,
        dstfds: np.ndarray,
        ransac_conf,
        checkr_conf,
        matches=None,
        T_gdth=None
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
    if len(matches) < 3:
        # useless matches
        return None
    ransac_conf.num_samples = min(len(matches), ransac_conf.num_samples)
    
    if matches is None:
        matches = init_matches(srcfds, dstfds)

    # build search tree on dst feature points
    dst_search_tree = o3d.geometry.KDTreeFlann(npy2o3d(dstkts))

    initial_T = None
    # infinite generator
    proposal_generator_serial = (matches[np.random.choice(range(matches.shape[0]), ransac_conf.num_samples, replace=False)] for _ in iter(int, 1))
    validator = lambda proposal: one_iter_match(srckts, dstkts, proposal, checkr_conf)

    # concurrently find the inital T, using 'with' method
    # so that Executor.shutdown(wait=True) is not needed.
    # log_dbug("finding initial transformation T...")
    t1 = time.time()
    try_limit = int(1e4)
    try_times = 0
    fall_back = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=ransac_conf.num_workers) as executor:
        Ts = []
        while len(Ts) < 2:
            if ransac_conf.num_samples - fall_back < 3:
                return None
            if try_times == try_limit:
                try_times = 0
                try_limit = try_limit // 2
                fall_back += 1

            proposal_generator_parall = (matches[np.random.choice(range(matches.shape[0]), ransac_conf.num_samples - fall_back, replace=False)] for _ in range(ransac_conf.num_workers))
            futures = [executor.submit(validator, proposal) for proposal in proposal_generator_parall]
            for future in concurrent.futures.as_completed(futures):
                try:
                    T = future.result()
                except Exception as e:
                    raise e
                if T is not None:
                    Ts.append(T)
                    break
            
            try_times += 1
    t2 = time.time()
    log_info(f"finding coarse candidate T costs {t2-t1:.2f}s with {ransac_conf.num_samples - fall_back} pairs")
    
    initial_T = None
    best_fitness = 0.0
    for T in Ts:
        candidate_evaluation = icp.ICP_exact_match(
            srckts, dstkts, dst_search_tree,
            T,
            ransac_conf.max_corrdist, ransac_conf.num_rfne
        )
        if candidate_evaluation.fitness >= best_fitness:
            best_fitness = candidate_evaluation.fitness
            initial_T = T

    # baseline
    best_evaluation = icp.ICP_exact_match(
        srckts, dstkts, dst_search_tree,
        initial_T,
        ransac_conf.max_corrdist, ransac_conf.num_rfne
    )
    num_validation = 0
    for _ in tqdm(range(ransac_conf.num_iter), total=ransac_conf.num_iter, ncols=100, desc="ICP refining"):
        T = validator(next(proposal_generator_serial))
        # check validity
        if T is not None and num_validation < ransac_conf.num_vald:
            curr_evaluation = icp.ICP_exact_match(
                srckts, dstkts, dst_search_tree,
                T,
                ransac_conf.max_corrdist, ransac_conf.num_rfne
            )
            if curr_evaluation.fitness > best_evaluation.fitness:
                best_evaluation = curr_evaluation
            num_validation += 1
        
        if num_validation == ransac_conf.num_vald:
            break

    return best_evaluation
