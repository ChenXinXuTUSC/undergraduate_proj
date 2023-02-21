import os
import heapq
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement
import open3d as o3d
from tqdm import tqdm

from . import colorlog
from .colorlog import *

def npz2ply(npz_path:str, overwrite_rgb:bool=False, new_rgb=None):
    if not os.path.exists(npz_path):
        raise Exception("npz file not exists, please check the path")
    if overwrite_rgb:
        if len(new_rgb) ^ 3:
            raise Exception("invalid rgb length")
    
    apcd = np.load(npz_path)
    # print(apcd.files)

    all_pos = apcd["pcd"]
    all_rgb = apcd["color"] * 255.0
    if overwrite_rgb:
        all_rgb[:,:] = new_rgb

    # print("pos:", all_pos.shape)
    # print("rgb:", all_rgb.shape)
    # print("minimum coord:", np.min(all_pos, axis=0))
    # print("maximum corrd:", np.max(all_pos, axis=0))

    return np.concatenate([all_pos, all_rgb], axis=1)

def npy2o3d(points:np.ndarray):
    if len(points.shape) > 2:
        raise Exception("npy2o3d could only handle data shape of [num_points, num_features]")
    
    points_o3d = o3d.geometry.PointCloud()
    if points.shape[1] >= 3:
        # for xyz coordinates
        points_o3d.points = o3d.utility.Vector3dVector(points[:, 0:3])
    if points.shape[1] >= 6:
        # for rgb colors
        points_o3d.colors = o3d.utility.Vector3dVector(points[:, 3:6])
    if points.shape[1]>= 9:
        # for uvw normals
        points_o3d.normals = o3d.utility.Vector3dVector(points[:, 6:9])
    
    return points_o3d

def o3d2npy(points_o3d:o3d.geometry.PointCloud):
    if len(points_o3d.points) > 0:
        xyz = np.asarray(points_o3d.points)
    if len(points_o3d.colors) > 0:
        rgb = np.asarray(points_o3d.colors)
    if len(points_o3d.normals) > 0:
        uvw = np.asarray(points_o3d.normals)
    return np.concatenate([xyz, rgb, uvw], axis=1)

def transform_augment(points: np.ndarray, angle: float, dist: float):
    ''' 
    Randomly rotate the point clouds along the Y-aixs\
    and randomly translate to augment the dataset.
    
    params:
    * points - NxF array, [num_points, num_features]
    * angle - maximum angle of random rotation, in degree
    * dist - maximum distance of random direction translation
    
    return:
    * points - NxF array, [num_points, num_features]
    '''
    # for semantic segmentation, F stands for 9 features
    rotation_angle = np.random.uniform() * (angle / 180.0) * np.pi # convert to pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotmat = np.array(
        [
            [cosval,  0.0, sinval],
            [0.0,     1.0,    0.0],
            [-sinval, 0.0, cosval]
        ]
    )
    transd = np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()]) * dist
    points[:, :3] = np.dot(points[:, :3], rotmat)[:, :3] + transd
    # 如果该点云已经计算了法向量，那么法向量也要进行变换
    if points.shape[1] >= 9:
        # 该函数只能假设点云的特征排列是(x,y,z,r,g,b,u,v,w)
        # 即最后三个位置存放法向量方向
        points[:, 6:9] = np.dot(points[:, 6:9], rotmat)[:, :3]
    return points, rotmat, transd

def voxel_down_sample(points: np.ndarray, voxel_size: float):
    '''
    makes sure that the first 3 dimension of the points arr\n
    is (x,y,z) coordinates
    '''
    max_coord = np.max(points[:,0:3], axis=0)
    min_coord = np.min(points[:,0:3], axis=0)
    # use numpy broadcast to do the calculation
    num_grids = (max_coord - min_coord) // voxel_size + 1

    voxel_indices = (points[:,:3] - min_coord) // voxel_size
    voxel_indices = voxel_indices[:,0] + voxel_indices[:,1]*num_grids[0] + voxel_indices[:,2]*num_grids[0]*num_grids[1]
    sorted_indices = np.argsort(voxel_indices) # 返回当元素按升序排序时对应位置的元素在原数组中的下标
    _, unique_indices = np.unique(voxel_indices[sorted_indices], return_index=True) # 重复元素在原数组第一次出现的下标
    unique_indices = np.append(unique_indices, len(points))
    
    filtered_points = np.zeros((0,6))
    for i in tqdm(range(len(unique_indices) - 1), total=len(unique_indices) - 1, ncols=100, desc="filtering"):
        filtered_points = np.append(filtered_points, [np.mean(points[sorted_indices[unique_indices[i]:unique_indices[i+1]]], axis=0)], axis=0)
    
    log_info(f"original num: {len(points)}, after voxel down sample: {len(filtered_points)}")
    return filtered_points

def fuse2frags(points1, points2, ply_line_type:np.dtype, out_dir:str=".", out_name:str="out.ply"):
    # ply_line_type = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    points1_plyformat = np.array([tuple(line) for line in points1], dtype=ply_line_type)
    points2_plyformat = np.array([tuple(line) for line in points2], dtype=ply_line_type)
    points = np.concatenate([points1_plyformat, points2_plyformat], axis=0)
    PlyData([PlyElement.describe(points, "vertex", comments="vertices")]).write(os.path.join(out_dir, out_name))

def solve_procrustes(P,Q):
    '''
    params
    -
    * P: [num_points, xyz] points array
    * Q: [num_points, xyz] points array

    return
    -
    * T: estimated transformation SE(3)
    '''
    # 泛化普鲁克分析默认需要分析的T只是一个旋转矩阵
    # 所以我们先把点集P和Q中心化，这样就不用求平移
    # 项t了。而且t在最小二乘法中求导很容易得出就是
    # P - QR，因此我们先求出R就可以得到t

    # 关于R为什么是使用SVD分解可以求，具体请参考这本书
    # Generalized Procrustes analysis and its applications in photogrammetry - Akca, Devrim
    # 利用了P*Q是一个对称矩阵的特性，推导出了很多重要的等式，最后发现就是SVD分解的两个特征向量矩阵的乘积
    
    # we use 'u' to represent greek 'Mu'
    # and 's' to represent greek 'sigma'
    Pu = P.mean(axis=0)
    Qu = Q.mean(axis=0)

    U, S, V = np.linalg.svd(np.dot(Qu.T, Pu), full_matrices=True, compute_uv=True)
    R = np.dot(U, V)
    t = Qu - np.dot(Pu, R)

    T = np.zeros((4, 4))
    T[0:3, :3] = R
    T[0:3, 3:] = t
    T[3, 3] = 1.0
    return T