import os
import copy
import numpy as np
from plyfile import PlyData, PlyElement
import open3d as o3d

from .colorlog import *

ply_edg_i1i2rgb = np.dtype(
    [
        ("vertex1", "int32"), 
        ("vertex2", "int32"),
        ("red", "u1"), 
        ("green", "u1"),
        ("blue", "u1")
    ]
)

def make_ply_vtx_type(has_rgb: bool=False, has_normal: bool=False):
    vtx_type = [
        ("x", "f4"), 
        ("y", "f4"),
        ("z", "f4")
    ]
    if has_rgb:
        vtx_type.extend(
            [
                ("red", "u1"), 
                ("green", "u1"), 
                ("blue", "u1"),
            ]
        )
    if has_normal:
        vtx_type.extend(
            [
                ("nx", "f4"),
                ("ny", "f4"),
                ("nz", "f4")
            ]
        )
    return np.dtype(vtx_type)

def npz2npy(npz_path: str, overwrite_rgb: bool=False, new_rgb=None):
    '''
    这个函数是针对3DMatch-FCGF的npz数据文件读取设计的，并不通用，对于
    其他npz存储文件，并不能提前知道里面存储了哪些属性，但是针对当前的
    数据集，是已知xyz坐标与rgb颜色属性的。
    '''
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

def npy2o3d(points: np.ndarray):
    if len(points.shape) > 2:
        raise Exception("npy2o3d could only handle data shape of [num_points, num_features]")
    
    points_o3d = o3d.geometry.PointCloud()
    if points.shape[1] >= 3:
        # for xyz coordinates
        points_o3d.points = o3d.utility.Vector3dVector(points[:, 0:3])
    if points.shape[1] >= 6:
        # for rgb colors
        points_o3d.colors = o3d.utility.Vector3dVector(points[:, 3:6])
    if points.shape[1] >= 9:
        # for uvw normals
        points_o3d.normals = o3d.utility.Vector3dVector(points[:, 6:9])
    
    return points_o3d

def o3d2npy(points_o3d: o3d.geometry.PointCloud):
    attr = []
    if len(points_o3d.points) > 0:
        attr.append(np.asarray(points_o3d.points)) 
    if len(points_o3d.colors) > 0:
        attr.append(np.asarray(points_o3d.colors))
    if len(points_o3d.normals) > 0:
        attr.append(np.asarray(points_o3d.normals))
    return np.concatenate(attr, axis=1)

def ply2npy(ply_path: str):
    try:
        plydata = PlyData.read(ply_path)
        element_names = [e.name for e in plydata.elements]
        if "vertex" not in element_names:
            raise Exception("no verticies data found, vertex data should be an element named \"vertex\"")
        # we only extract verticies data
        points = np.asarray(plydata["vertex"].data)
        if len(points) == 0:
            raise Exception("empty vertex data")
        
        npy_points = []
        for line in points:
            npy_points.append(list(line))
        npy_points = np.asarray(npy_points)
    except Exception as e:
        log_erro(str(e))
        return None
    return npy_points

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
        points[:, 6:9] = np.dot(points[:, 6:9], rotmat)
    return points, rotmat, transd

def build_random_transform(angle: float, dist: float):
    rotation_angle = (angle + 10.0 * (np.random.uniform() - 0.5)) / 180.0 * np.pi # convert to pi
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
    T = np.eye(4)
    T[:3,:3] = rotmat
    T[:3, 3] = transd
    return T

def voxel_down_sample(points: np.ndarray, voxel_size: float):
    '''compute the down sampled points' coordinates
    
    params
    -
    * points: np.ndarray.
        Original dense point cloud data in shape
        (num_pts, num_feats).
    * voxel_size: float.
        Unit size of down sample operation,  not
        radius but length of cubic.
    
    return
    -
    * down-sampled coords: np.ndarray.
        Down sampled points coordinates in shape
        (num_pts, num_feats).
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
    
    filtered_points = np.zeros((0, points.shape[1]))
    for i in range(len(unique_indices) - 1):
        filtered_points = np.append(filtered_points, [np.mean(points[sorted_indices[unique_indices[i]:unique_indices[i+1]]], axis=0)], axis=0)
    
    log_info(f"voxel downsample: {len(points)}->{len(filtered_points)}")
    return filtered_points

def voxel_down_sample_gpt(points: np.ndarray, voxel_size: float, use_avg: bool=False):
    '''code given by ChatGPT, works fine
    
    params
    -
    * points: np.ndarray.
        Original dense point cloud data in shape
        (num_pts, num_feats).
    * voxel_size: float.
        Unit size of down sample operation,  not
        radius but length of cubic.
    * use_avg: bool.
        Whether to use the average of all coords
        in the same voxel or not. If false, choo
        se one point in the voxel.
    
    return
    -
    * down-sampled coords: np.ndarray.
        Down sampled points coordinates in shape
        (num_pts, num_feats).
    * quantized coords: np.ndarray.
        Quantized   coordinates   of  the  down-
        sampled ones. (int, int, int).
    * idx_dse2vox: np.ndarray.
        If 'use_avg' is true, indices  of  dense
        point cloud coords array to form the vox
        array is provided.
    '''
    # find the bounding box wrapping all points
    min_coord = np.min(points[:, :3], axis=0)
    max_coord = np.max(points[:, :3], axis=0)
    
    # calculate voxels of each axis
    voxel_numaxis = (max_coord - min_coord) // voxel_size
    
    # calculate voxel index of each point
    voxel_indices = (points[:, :3] - min_coord) // voxel_size
    
    # calculate voxel center of each point
    voxel_centers = voxel_indices * voxel_size + voxel_size / 2.0
    
    # group the points by voxel indices
    # return_index=True
    #   given a list of subscript which can be used to choose elements
    #   from the original array to form the unique array
    # return_inverse=True
    #   given a list of subscript which can be used to choose elements
    #   from the unique array to form the original array
    voxel_uniques, idx_old2new, idx_new2old, unique_counts = np.unique(
        voxel_indices, axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True
    )
    
    voxel_points = np.zeros((len(voxel_uniques), points.shape[1]))
    
    if use_avg:
        np.add.at(voxel_points, idx_new2old, points)
        voxel_points /= unique_counts.reshape(-1, 1) # column divided by column
    else:
        voxel_points = points[idx_old2new]
    
    return voxel_points, voxel_uniques, None if use_avg else idx_old2new

def dump1frag(
        points: np.ndarray,
        ply_vertex_type: np.dtype,
        out_dir: str=".",
        out_name: str="out.ply"
    ):
    points = np.array([tuple(line) for line in points], dtype=ply_vertex_type)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, mode=0o755)
    PlyData(
        [
            PlyElement.describe(points, "vertex", comments="vertices")
        ]
    ).write(os.path.join(out_dir, out_name))

def fuse2frags(
        points1: np.ndarray, 
        points2: np.ndarray, 
        ply_vertex_type: np.dtype,
        out_dir: str=".", out_name: str="out.ply"
    ):
    points1_plyformat = np.array([tuple(line) for line in points1], dtype=ply_vertex_type)
    points2_plyformat = np.array([tuple(line) for line in points2], dtype=ply_vertex_type)
    points = np.concatenate([points1_plyformat, points2_plyformat], axis=0)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, mode=0o755)
    PlyData(
        [
            PlyElement.describe(points, "vertex", comments=["vertices"])
        ]
    ).write(os.path.join(out_dir, out_name))

def fusexfrags(
        points_list: list,
        ply_vertex_type: np.dtype,
        out_dir: str=".", out_name: str="out.ply"
    ):
    points_plyformat = []
    for points in points_list:
        for line in points:
            points_plyformat.append(tuple(line))
    points = np.array(points_plyformat, dtype=ply_vertex_type)
    PlyData(
        [
            PlyElement.describe(points, "vertex", comments=["vertices"])
        ]
    ).write(os.path.join(out_dir, out_name))

def fuse2frags_with_matches(
        points1: np.ndarray, 
        points2: np.ndarray, 
        matches: np.ndarray,
        ply_vertex_type: np.dtype,
        ply_line_type: np.dtype,
        out_dir: str=".", out_name: str="out.ply",
        correct: np.ndarray=None
    ):
    points1_plyformat = np.array([tuple(line) for line in points1], dtype=ply_vertex_type)
    points2_plyformat = np.array([tuple(line) for line in points2], dtype=ply_vertex_type)
    points = np.concatenate([points1_plyformat, points2_plyformat], axis=0)
    
    base_offset = len(points1)
    matches[:,1] += base_offset
    colors = np.ones((len(matches), 3))
    if correct is not None:
        colors[:, 0] *= 255.0
        colors[:, 1] *= correct.astype(np.float32)
        colors[:, 2]  = 0.0
    else:
        colors *= 255.0
    edges = np.concatenate([matches, colors], axis=1)
    edges = np.array([tuple(line) for line in edges], dtype=ply_line_type)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, mode=0o755)
    PlyData(
        [
            PlyElement.describe(points, "vertex", comments=["vertices"]),
            PlyElement.describe(edges, "edge", comments=["edges"])
        ]
    ).write(os.path.join(out_dir, out_name))

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

    # 保险起见，万一输入进来的点除了xyz坐标外还有其他的特征
    # 只使用xyz坐标进行普鲁克分析，以及闭式解的计算
    P = P[:, :3]
    Q = Q[:, :3]

    P_center = P.mean(axis=0)
    Q_center = Q.mean(axis=0)
    Pu = P - P_center # P decentralized
    Qu = Q - Q_center # Q decentralized

    U, S, V = np.linalg.svd(np.dot(Pu.T, Qu), full_matrices=True, compute_uv=True)
    R = np.dot(U, V)
    # t = Q_center - np.dot(P_center, R)
    t = np.mean(Q - P @ R, axis=0) 

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def apply_transformation(srcpts: np.ndarray, T: np.ndarray):
    import torch

    srcpts = copy.deepcopy(srcpts)
    if type(srcpts) == torch.Tensor:
        srcpts = srcpts.detach().numpy()
    if T.shape != (4, 4):
        log_warn("invalid transformation matrix")
        return srcpts
    R = T[:3, :3]
    t = T[:3, 3]
    if np.fabs(np.linalg.det(np.dot(R, R.T)) - 1.0) > 1e-3:
        log_warn("invalid rotation matrix, not orthogonal")
        return srcpts
    srcpts[:, :3] = srcpts[:, :3] @ R + t
    if srcpts.shape[1] >= 9:
        # 该函数只能假设点云的特征排列是(x,y,z,r,g,b,u,v,w)
        # 即最后三个位置存放法向量方向
        srcpts[:, 6:9] = srcpts[:, 6:9] @ R
    return srcpts

def resolve_axis_angle(T: np.ndarray, deg: bool):
    '''
    this function is referencing:\n
    https://blog.csdn.net/Sandy_WYM_/article/details/84309000
    '''
    if T.shape != (3, 3) and T.shape != (4, 4):
        log_warn("invalid rotation/transformation matrix")
        return None, None
    
    R = T
    if T.shape == (4, 4):
        R = T[:3,:3]
    angle = np.arccos((R.trace() - 1.0)/2.0)
    if deg:
        # if in degree format instead of radian
        angle = angle / np.pi * 180.0
    raxis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2.0 * np.sin(angle))
    raxis = raxis / np.sqrt((raxis * raxis).sum()) # normalization
    return raxis, angle

def ground_truth_matches(matches: np.ndarray, pcd1, pcd2, radius: float, T: np.ndarray):
    '''
    params
    ----------
    * matches: 2xN np.ndarray
    * pcd1: 3xN np.ndarray or o3d.geometry.PointCloud
    * pcd2: 3xN np.ndarray or o3d.geometry.PointCloud
    * radius: radius for outlier matches
    * T: 4x4 homogeneous transformation if pcd1 and pcd2 is not aligned

    return
    ----------
    * is_correct: 1XN boolean np.ndarray that indicates inlier matches
    '''
    import torch
    pcd1 = copy.deepcopy(pcd1)
    pcd2 = copy.deepcopy(pcd2)
    if type(pcd1) == o3d.geometry.PointCloud:
        pcd1 = o3d2npy(pcd1)
    if type(pcd2) == o3d.geometry.PointCloud:
        pcd2 = o3d2npy(pcd2)

    if type(pcd1) == torch.Tensor:
        pcd1 = pcd1.detach().numpy()
    if type(pcd2) == torch.Tensor:
        pcd2 = pcd2.detach().numpy()
    
    pcd1 = pcd1[:, :3]
    pcd2 = pcd2[:, :3]

    if T is not None:
        pcd1 = apply_transformation(pcd1, T)
    
    is_correct = ((pcd1[matches[:, 0]] - pcd2[matches[:, 1]]) ** 2).sum(axis=1) < radius ** 2
    # log_dbug("in filter:\n", ((pcd1[matches[is_correct][:, 0]] - pcd2[matches[is_correct][:, 1]]) ** 2).sum(axis=1))
    return is_correct

def principle_K_components(samples: np.ndarray, k: int):
    '''
    params
    ----------
    * samples-np.ndarray: each line represents a sample data

    return
    ----------
    * principle component variable columns indices in decreasing order
    '''
    samples = samples - samples.mean(axis=0)
    cov = np.matmul(samples.T, samples) / len(samples)
    # numpy.linalg.eig() return eigen values and eigen
    # vectors, denoted by w, v, where each  column  of
    # the v is an eigen vector(that means  you  should
    # use v[:, idx] to access one eigen vector). Eigen
    # vectors are orthogonal to each other and are nor
    # malized to unit vector
    eigvals, eigvecs = np.linalg.eig(cov)
    # numpy.argsort return indices that would sort
    # the original array in ascend order
    eigvecs = eigvecs[:,eigvals.argsort()[::-1]][:, :k] # flip into descend order

    return eigvecs

def dump_registration_result(
    out_dir: str,
    out_name: str,
    points1: np.ndarray,
    points2: np.ndarray,
    downsampled_coords1: np.ndarray, keyptsidx1: np.ndarray,
    downsampled_coords2: np.ndarray, keyptsidx2: np.ndarray,
    T_gdth,
    T_pred,
    gdth_matches=None
):
    # 给关键点上亮色，请放在其他点上色完成后再给关键点上色，否则关键点颜色会被覆盖
    downsampled_coords1[keyptsidx1, 3:6] = np.array([255, 0, 0])
    downsampled_coords2[keyptsidx2, 3:6] = np.array([0, 255, 0])
    # show matches
    if gdth_matches is not None:
        fuse2frags_with_matches(
            apply_transformation(downsampled_coords1, T_pred), downsampled_coords2, 
            gdth_matches, make_ply_vtx_type(True, True), 
            ply_edg_i1i2rgb,
            f"{out_dir}/matches", f"{out_name}_matches.ply"
        )
    
    # contrastive comparison
    points1[:,3:6] = [0, 255, 255]
    points2[:,3:6] = [255, 255, 0]
    fuse2frags(
        apply_transformation(points1, np.eye(4)), points2, 
        make_ply_vtx_type(True, True), f"{out_dir}/orgl_contrastive", f"{out_name}_orgl.ply"
    )
    fuse2frags(
        apply_transformation(points1, T_pred), points2, 
        make_ply_vtx_type(True, True), f"{out_dir}/pred_contrastive", f"{out_name}_pred.ply"
    )
    fuse2frags(
        apply_transformation(points1, T_gdth), points2, 
        make_ply_vtx_type(True, True), f"{out_dir}/gdth_contrastive", f"{out_name}_gdth.ply"
    )
