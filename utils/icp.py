import numpy as np
import open3d as o3d

from . import colorlog
from .colorlog import *
from . import tools
from .tools import *

def early_terminate(curr_res, prev_res):
    relative_fitness_gain = curr_res.fitness / prev_res.fitness - 1.0
    return relative_fitness_gain < 1e-2

def ICP_exact_match(
        srcpts: np.ndarray,
        dstpts: np.ndarray, dst_search_tree,
        T: np.ndarray,
        max_corresponding_dist, max_iter_num
    ):
    # in case that point line contains features other than xyz coordinate
    srcpts = srcpts[:, :3]
    dstpts = dstpts[:, :3]

    prev_res = o3d.pipelines.registration.evaluate_registration(
        srcpts, dstpts, max_corresponding_dist, T
    )
    curr_res = prev_res

    # 我们就不在ICP里面再重复建立KD树了，这个重复的任务只需要在函数外
    # 完成一次即可，作为参数传进来吧，虽然函数的参数列表就没有那么整齐
    # dst_search_tree = o3d.geometry.KDTreeFlann(npy2o3d(dstpts))
    for _ in range(max_iter_num):
        pts_transed = np.dot(srcpts, T)
        matches = []
        for query_idx, query_pt in enumerate(pts_transed):
            # no need to check if result is None
            # 注意o3d的KNN搜索和RNN搜索返回结果的不同，RNN进行半径领域搜索，所以返回的第一个点就是自身
            # 而KNN搜索返回的结果中不会包含自身，
            _, neighbors_indicies, neighbors_dists = dst_search_tree.search_knn_vector_3d(query_pt, 1)
            if neighbors_dists[0] <= max_corresponding_dist:
                matches.append([query_idx, neighbors_indicies[0]])
        matches = np.asarray(matches)

        # ICP
        if len(matches) >= 4:
            T = solve_procrustes(pts_transed, dstpts)
            curr_res = o3d.pipelines.evaluate_registration(srcpts, dstpts, max_corresponding_dist, T)
            if early_terminate(curr_res, prev_res):
                log_info("early stopping the RANSAC ICP procedure")
                break
    return curr_res

