import os
import numpy as np
from plyfile import PlyData, PlyElement
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

def fuse2frags(points1, points2, out_name:str):
    ply_line_type = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    points1_plyformat = np.array([tuple(line) for line in points1], dtype=ply_line_type)
    points2_plyformat = np.array([tuple(line) for line in points2], dtype=ply_line_type)
    points = np.concatenate([points1_plyformat, points2_plyformat], axis=0)
    PlyData([PlyElement.describe(points, "vertex", comments="vertices with rgb")]).write("./augtsample/" + out_name + ".ply")

def iss_detect(points, search_tree, radius):
    '''
    Detect point cloud key points using Intrinsic Shape Signature(ISS)

    params
    ----------
    point_cloud: numpy.ndarray
        input point cloud
    search_tree: Open3D.geometry.KDTree
        point cloud search tree
    radius: float
        radius for ISS computing

    return
    ----------
    point_cloud: numpy.ndarray
        Velodyne measurements as N-by-3 numpy ndarray
    '''
    # please figure out the difference between k nearest neighbour
    # search and radius search. In this ISS key  points  detection
    # implementation, we use radius search provided by open3d.

    


