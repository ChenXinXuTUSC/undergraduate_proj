import os

import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement

import config
import utils


def npz2npy(npz_path:str, overwrite_rgb:bool=False, new_rgb=None):
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
    
    utils.log_info(f"original num: {len(points)}, after voxel down sample: {len(filtered_points)}")
    return filtered_points

def fuse2frags(points1, points2, out_name:str):
    ply_line_type = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    points1_plyformat = np.array([tuple(line) for line in points1], dtype=ply_line_type)
    points2_plyformat = np.array([tuple(line) for line in points2], dtype=ply_line_type)
    points = np.concatenate([points1_plyformat, points2_plyformat], axis=0)
    PlyData([PlyElement.describe(points, "vertex", comments="vertices with rgb")]).write("./fusesample/" + out_name + ".ply")


if __name__ == "__main__":
    args = vars(config.args)

    npzs = [os.path.join(args["3dmatch_root"], "npz", file) for file in sorted(os.listdir(os.path.join(args["3dmatch_root"], "npz")))]
    txts = [os.path.join(args["3dmatch_root"], "txt", file) for file in sorted(os.listdir(os.path.join(args["3dmatch_root"], "txt")))]

    # for txt in txts:
    #     print(txt)

    # generate some demo pointcloud
    with open(txts[0], 'r') as f:
        lines = f.readlines()
        lines = [line.rstrip().split(' ') for line in lines]
        for line in lines:
            # , overwrite_rgb=True, new_rgb=[255, 0, 0]
            # , overwrite_rgb=True, new_rgb=[0, 255, 0]
            points1 = npz2npy(os.path.join(args["3dmatch_root"], "npz", line[0]))
            points2 = npz2npy(os.path.join(args["3dmatch_root"], "npz", line[1]))
            # points1 = voxel_down_sample(points1, 0.05)
            # points2 = voxel_down_sample(points2, 0.05)
            # print(f"{line[0]}:{points1.shape}, {line[1]}:{points2.shape}, overlap ratio:{line[2]}")
            fuse2frags(points1, points2, line[0].split('.')[0]+'_'+line[1].split('.')[0].split('@')[1]+'_'+line[2])
