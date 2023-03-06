import heapq
import numpy as np
import pandas as pd
import open3d as o3d

from . import colorlog
from .colorlog import *
from . import tools
from .tools import *


def radius_outlier_filter(points:np.ndarray, radius:float, must_neighbors:int):
    '''
    params
    -
    * points: [num_points, xyz] points array
    * radius: search radius from the referenced point
    * must_neighbors: number of neighbors in search result can\n
    not be less than this value

    return
    -
    * points: [num_points, xyz] points array filtered
    '''
    kdtree = o3d.geometry.KDTreeFlann(npy2o3d(points))
    pts_idx_after = []
    num_points = points.shape[0]
    for i in range(num_points):
        num_neighbors, pts_idx, _ = kdtree.search_radius_vector_3d(points[i,:3], radius)
        if num_neighbors - 1 > must_neighbors:
            pts_idx_after.append(i)
    return points[pts_idx_after]

def iss_detect(points: np.ndarray, radius=0.25):
    '''
    Detect point cloud key points using Intrinsic Shape Signature(ISS)\n
    ISS角点检测原论文要求使用RNN进行球形领域搜索，所以就不提供KNN的选项了\n
    不要使用open3d的PointCloud对象来做搜索，角点检测的似乎不是非常正确，比\n
    如很多平面上本来不应该有角点，但仍然均匀分布了角点

    params
    ----------
    * point_cloud: numpy.ndarray
    * search_scheme: KNN or RNN oct-tree search pattern
    * r: radius used for RNN
    * k: number of neighbours used for KNN

    return
    ----------
    * point_cloud: numpy.ndarray
    '''
    # please figure out the difference between k nearest neighbour
    # search and radius search. In this ISS key  points  detection
    # implementation, we use radius search provided by open3d.

    # 准备好字典键序列，等会转成data sheet表格
    # 这就类似于argparse的参数传递出去之后无法
    # 使用成员运算符访问，必须用easydict这个包
    # 重新构建。通过转为表格的形式， 我们可以根
    # 据某一列作为键值而对行进行排序，这样相关
    # 属性放在同一行就可以作为整体一起被移动到
    # 对应的顺位
    keypoints = {
        "id": [],
        "x": [],
        "y": [],
        "z": [],
        "eigval_1": [], # 特征值1
        "eigval_2": [], # 特征值2
        "eigval_3": []  # 特征值3
    }

    # first construct a search tree
    search_tree = o3d.geometry.KDTreeFlann(npy2o3d(points))
    num_neighbors_cache = {}
    little_heap = []
    for center_idx, center in enumerate(points):
        num_neighbors, neighbor_indicies, _ = search_tree.search_radius_vector_3d(center[:3], radius)
        if num_neighbors < 6:
            continue # heuristic to filter outlier
        weights = []
        distans = []
        # 因为用的是RNN搜索，所以要去除第一个自身索引
        for neighbor_idx in neighbor_indicies[1:]:
            # check if in the cache
            if not neighbor_idx in num_neighbors_cache:
                neigneig_num, _, _ = search_tree.search_radius_vector_3d(points[neighbor_idx, :3], radius)
                num_neighbors_cache[neighbor_idx] = neigneig_num - 1 if neigneig_num > 1 else 1
            weights.append(1.0/num_neighbors_cache[neighbor_idx])
            distans.append(points[neighbor_idx, :3] - center[:3])
        weights = np.array(weights)
        distans = np.array(distans)
        # 这里的协方差矩阵指的是变量维度之间的协方差，不是样本之间的协方差
        # 因为只有坐标(x,y,z)三个维度，那么三个变量维度的协方差矩阵就是3x3的
        covariance = np.dot(distans.T, np.dot(np.diag(weights), distans)) / weights.sum() # ??? 3x3 instead of nxn
        eigval, eigvec = np.linalg.eig(covariance)
        eigval = eigval[eigval.argsort()[::-1]] # 降序排序，原argsort是返回从小到大的元素索引
        
        # 三个特征值中的最大特征值作为排序依据，索引加入到小根堆
        heapq.heappush(little_heap, (-eigval[2], center_idx))

        # add to dataframe
        keypoints["id"].append(center_idx)
        keypoints["x"].append(center[0])
        keypoints["y"].append(center[1])
        keypoints["z"].append(center[2])
        keypoints["eigval_1"].append(eigval[0])
        keypoints["eigval_2"].append(eigval[1])
        keypoints["eigval_3"].append(eigval[2])
    
    # 非极大值抑制，并不是每个点都需要成为关键点
    suppressed_points_indicies = set()
    while little_heap:
        _, top_idx = heapq.heappop(little_heap)
        if not top_idx in suppressed_points_indicies:
            _, top_neighbors_indices, _ = search_tree.search_radius_vector_3d(points[top_idx, :3], radius)
            for idx in top_neighbors_indices[1:]:
                suppressed_points_indicies.add(idx)
    
    # 格式化为data frame好进行整体关联操作
    keypoints = pd.DataFrame.from_dict(keypoints)
    keypoints = keypoints.loc[
        keypoints["id"].apply(lambda idx: not idx in suppressed_points_indicies),
        keypoints.columns
    ]

    # 协方差矩阵的特征值可以看作 参与协方差运算的所有样本，在
    # 该特征值对应的向量方向上的分量平方之和，特征值越大， 在
    # 该方向上的分布越发散。

    # e1=e2>e3时，近邻点在两个方向上散布均匀，则近邻点拟合了
    # 一个较扁的椭球，类似于一个平面，平面不是一个好的特征。
    
    # e1>e2=e3时，近邻点在主方向上的分布均匀，但剩下两个方向
    # 分布致密，类似于细椭球，就像标枪一样，类似于一条直线。
    # 而直线也不是一个好的特征。

    # eigval3_threshold = np.median(keypoints["eigval_3"].values)
    keypoints = keypoints.loc[
        (keypoints["eigval_1"] / keypoints["eigval_2"] > 1.5) &
        (keypoints["eigval_2"] / keypoints["eigval_3"] > 1.5),
        # keypoints["eigval_3"] > eigval3_threshold,
        keypoints.columns
    ]

    # return the keypoints in decreasing eigen value order
    keypoints = keypoints.sort_values("eigval_3", axis=0, ascending=False, ignore_index=True)
    return keypoints

def iss_detect_copy(pcd: o3d.geometry.PointCloud, radius: float):
    '''
    Detect point cloud key points using Intrinsic Shape Signature(ISS)

    Parameters
    ----------
    point_cloud: Open3D.geometry.PointCloud
        input point cloud
    search_tree: Open3D.geometry.KDTree
        point cloud search tree
    radius: float
        radius for ISS computing

    Returns
    ----------
    point_cloud: numpy.ndarray
        Velodyne measurements as N-by-3 numpy ndarray
    '''

    # points handler:
    points = np.asarray(pcd.points)
    search_tree = o3d.geometry.KDTreeFlann(pcd)
    # keypoints container:
    keypoints = {
        'id': [],
        'x': [],
        'y': [],
        'z': [],
        'lambda_0': [],
        'lambda_1': [],
        'lambda_2': []
    }

    # cache for number of radius nearest neighbors:
    num_rnn_cache = {}
    # heapq for non-maximum suppression:
    pq = []
    for idx_center, center in enumerate(
        points
    ):
        # find radius nearest neighbors:
        [k, idx_neighbors, _] = search_tree.search_radius_vector_3d(center, radius)

        # use the heuristics from NDT to filter outliers:
        if k < 6:
            continue

        # for each point get its nearest neighbors count:
        w = []
        deviation = []
        for idx_neighbor in np.asarray(idx_neighbors[1:]):
            # check cache:
            if not idx_neighbor in num_rnn_cache:
                [k_, _, _] = search_tree.search_radius_vector_3d(
                    points[idx_neighbor], 
                    radius
                )
                num_rnn_cache[idx_neighbor] = k_
            # update:
            w.append(num_rnn_cache[idx_neighbor])
            deviation.append(points[idx_neighbor] - center)
        
        # calculate covariance matrix:
        w = np.asarray(w)
        deviation = np.asarray(deviation)

        cov = (1.0 / w.sum()) * np.dot(
            deviation.T,
            np.dot(np.diag(w), deviation)
        )

        # get eigenvalues:
        w, _ = np.linalg.eig(cov)
        w = w[w.argsort()[::-1]]

        # add to pq:
        heapq.heappush(pq, (-w[2], idx_center))

        # add to dataframe:
        keypoints['id'].append(idx_center)
        keypoints['x'].append(center[0])
        keypoints['y'].append(center[1])
        keypoints['z'].append(center[2])
        keypoints['lambda_0'].append(w[0])
        keypoints['lambda_1'].append(w[1])
        keypoints['lambda_2'].append(w[2])
    
    # non-maximum suppression:
    suppressed = set()
    while pq:
        _, idx_center = heapq.heappop(pq)
        if not idx_center in suppressed:
            # suppress its neighbors:
            [_, idx_neighbors, _] = search_tree.search_radius_vector_3d(
                points[idx_center], 
                radius
            )
            for idx_neighbor in np.asarray(idx_neighbors[1:]):
                suppressed.add(idx_neighbor)
        else:
            continue

    # format:        
    keypoints = pd.DataFrame.from_dict(
        keypoints
    )

    # first apply non-maximum suppression:
    keypoints = keypoints.loc[
        keypoints['id'].apply(lambda id: not id in suppressed),
        keypoints.columns
    ]

    # then apply decreasing ratio test:
    keypoints = keypoints.loc[
        (keypoints['lambda_0'] > keypoints['lambda_1']) &
        (keypoints['lambda_1'] > keypoints['lambda_2']),
        keypoints.columns
    ]

    # finally, return the keypoint in decreasing lambda 2 order:
    keypoints = keypoints.sort_values('lambda_2', axis=0, ascending=False, ignore_index=True)
    
    return keypoints
