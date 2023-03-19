import numpy as np
import open3d as o3d
from tqdm import tqdm
import copy

from . import colorlog
from .colorlog import *
from . import tools
from .tools import *

def early_terminate(curr_res, prev_res):
    relative_fitness_gain = (curr_res.fitness+1.0) / (prev_res.fitness+1.0) - 1.0
    return relative_fitness_gain < 1e-2

def ICP_exact_match(
        srcpts: np.ndarray,
        dstpts: np.ndarray, dst_search_tree,
        T: np.ndarray,
        max_corresponding_dist, max_iter_num
    ):
    # in case that point line contains features other than xyz coordinate
    srcpts = copy.deepcopy(srcpts[:, :3])
    dstpts = copy.deepcopy(dstpts[:, :3])

    # the class <evaluate_registration> has property <correspondence set>
    # that is an nx2 numpy int-type ndarray, it stores the corresponding
    # index of src and dst feature. It also has a <transform> proerty, it
    # is a 4x4 numpy float ndarray that stores the SE(3)
    prev_res = o3d.pipelines.registration.evaluate_registration(
        npy2o3d(srcpts), npy2o3d(dstpts), max_corresponding_dist, T
    )
    curr_res = prev_res

    # 我们就不在ICP里面再重复建立KD树了，这个重复的任务只需要在函数外
    # 完成一次即可，作为参数传进来吧，虽然函数的参数列表就没有那么整齐
    # dst_search_tree = o3d.geometry.KDTreeFlann(npy2o3d(dstpts))
    for _ in range(max_iter_num):
        pts_transed = np.dot(srcpts, T[:3, :3]) + T[:3, 3]
        matches = []
        for query_idx, query_kpt in enumerate(pts_transed):
            # no need to check if result is None
            # 注意o3d的KNN搜索和RNN搜索返回结果的不同，RNN进行半径领域搜索，所以返回的第一个点就是自身
            # 而KNN搜索返回的结果中不会包含自身，
            _, neighbors_indicies, neighbors_dists = dst_search_tree.search_knn_vector_3d(query_kpt, 1)
            if neighbors_dists[0] <= max_corresponding_dist:
                matches.append([query_idx, neighbors_indicies[0]])
        matches = np.asarray(matches)

        # ICP
        if len(matches) >= 4:
            P = srcpts[matches[:, 0]]
            Q = dstpts[matches[:, 1]]
            T = solve_procrustes(P, Q)
            curr_res = o3d.pipelines.registration.evaluate_registration(npy2o3d(srcpts), npy2o3d(dstpts), max_corresponding_dist, T)
            if early_terminate(curr_res, prev_res):
                tqdm.write(log_info("early stopping the RANSAC ICP procedure", quiet=True))
                break
    return curr_res

def ICP_exact_match_copy(
        source_pcd, target_pcd, search_tree_target,
        T,
        max_correspondence_dist, max_iteration
):
    # num. points in the source:
    N = len(source_pcd.points)

    # evaluate relative change for early stopping:
    result_prev = result_curr = o3d.pipelines.registration.evaluate_registration(
        source_pcd, target_pcd, max_correspondence_dist, T
    )

    for _ in range(max_iteration):
        # transform is actually an in-place operation. deep copy first otherwise the result will be WRONG
        source_pcd_copy = copy.deepcopy(source_pcd)
        # apply transform:
        source_pcd_copy = source_pcd_copy.transform(T)

        # find correspondence:
        matches = []
        for n in range(N):
            query = np.asarray(source_pcd_copy.points)[n]
            _, idx_nn_target, dis_nn_target = search_tree_target.search_knn_vector_3d(query, 1)

            if dis_nn_target[0] <= max_correspondence_dist:
                matches.append([n, idx_nn_target[0]])
        matches = np.asarray(matches)

        if len(matches) >= 4:
            # sovle ICP:
            P = np.asarray(source_pcd.points)[matches[:, 0]]
            Q = np.asarray(target_pcd.points)[matches[:, 1]]
            T = solve_procrustes(P, Q)

            # evaluate:
            result_curr = o3d.pipelines.registration.evaluate_registration(
                source_pcd, target_pcd, max_correspondence_dist, T
            )

            # if no significant improvement:提前中止
            if early_terminate(result_curr, result_prev):
                break

    return result_curr
