import os
from os import path as osp
import numpy as np
from tqdm import tqdm
import copy

import utils

import cv2

class PairDataset:
    '''
    <PairDataset> is a base class, not for data itering.
    Class  derived  from  <PairDataset>   will    handle
    different input source, and return two point  clouds
    when itered by invoker(possibly  return  with  other 
    info)
    '''
    def __init__(
            self,
            root:str,
            shuffle: bool,
            augment: bool,
            augdgre: float,
            augdist: float
        ) -> None:
        self.root = os.path.abspath(root)
        self.shuffle = shuffle
        self.augment = augment
        self.augdgre = augdgre
        self.augdist = augdist

class ModelNet40Dense(PairDataset):
    def __init__(
            self, 
            root: str, 
            shuffle: bool, 
            augment: bool, 
            augdgre: float, 
            augdist: float,
            args=None
        ) -> None:
        super().__init__(root, shuffle, augment, augdgre, augdist)
        self.files = []
        self.classes = []
        self.partition = None

        if args is not None:
            self.classes = [cls.strip() for cls in args.classes.split()]
            if args.partition > 0.05:
                self.partition = args.partition
        mdirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        if len(self.classes) > 0:
            mdirs = [mdir for mdir in mdirs if mdir in self.classes]
        for mdir in tqdm(mdirs, total=len(mdirs), ncols=100, desc=self.__class__.__name__):
            plys = sorted([ply for ply in os.listdir(os.path.join(root, mdir)) if ply.endswith(".ply")])
            for ply in plys:
                self.files.append((mdir, ply[:-4], os.path.abspath(os.path.join(root, mdir, ply))))
        
        if shuffle:
            np.random.shuffle(self.files)
        
        self.iterate_pos = -1

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        _, sample_name, sample_path = self.files[idx]
        # # if sample is in txt format
        # with open(sample_path, 'r') as f:
        #     lines = f.readlines()
        #     lines = list(map(lambda li: [float(x) for x in li], [line.rstrip().split(',') for line in lines])) 
        # points = np.asarray(lines)

        points = utils.ply2npy(sample_path)
        # add dummy rgb attributes, as point clouds
        # in ModelNet40 dataset don't have colors.
        points = np.concatenate((points[:,0:3], np.zeros((len(points), 3)), points[:,3:6]), axis=1)
        
        if self.partition is not None:
            part1, part2, part_overlap = self.split_by_plane(points, 0.5)
        else:
            part1 = points
            part2 = copy.deepcopy(part1)

        T_gdth = np.eye(4)
        if self.augment:
            T_gdth = utils.build_random_transform(self.augdgre, self.augdist)
            part2 = utils.apply_transformation(part2, T_gdth)
        
        return part1, part2, T_gdth, sample_name

    def __next__(self):
        self.iterate_pos += 1
        if self.iterate_pos >= len(self.files):
            raise StopIteration
        return self[self.iterate_pos]

    def __iter__(self):
        return self

    def split_by_plane(self, points: np.ndarray, overlap_distance: float):
        '''split a point cloud by a plane with overlapping area.
        
        params
        -
        * points: np.ndarray.
            Points in shape(num_pts, feat_dimensions)
        * overlap_distance: float.
            Vertical distance of other point to plane
        
        return
        -
        * part1: np.ndarray.
            Points upon the plane.
        * part2: np.ndarray.
            Points below the plane.
        '''
        coords = points[:, :3] # only need xyz
        plane_point = coords.mean(axis=0)
        plane_normal = utils.principle_K_components(coords, 1).flatten()
        plane_normal /= np.linalg.norm(plane_normal)
        # Calculate the distance of each point in the point cloud to the plane  
        distances = np.dot(coords - plane_point, plane_normal)  
        
        # Split the point cloud into two separate arrays based on the sign of the distance  
        positive_points = points[distances >= -overlap_distance]  
        negative_points = points[distances <= +overlap_distance]  
        
        # Calculate the buffer zone around the plane  
        buffer_points = points[
            np.logical_and(
                distances >= -overlap_distance, 
                distances <= +overlap_distance
            )
        ]  
        
        return positive_points, negative_points, buffer_points

class ThreeDMatchFCGF(PairDataset):
    def __init__(
            self, 
            root: str, 
            shuffle: bool, 
            augment: bool, 
            augdgre: float, 
            augdist: float,
            args=None
        ) -> None:
        super().__init__(root, shuffle, augment, augdgre, augdist)
        self.files = []
        self.overlap_dn = 0.0
        self.overlap_up = 1.0
        self.rooms_included = []
        
        if args is not None:
            self.rooms_included = [room.strip() for room in args.rooms.split()]
            self.overlap_dn = args.overlap_dn
            self.overlap_up = args.overlap_up
        # npzs = [os.path.join(root, "npz", file) for file in sorted(os.listdir(os.path.join(root, "npz")))]
        txts = [file for file in sorted(os.listdir(os.path.join(self.root, "txt")))]
        for txt in tqdm(txts, total=len(txts), ncols=100, desc=self.__class__.__name__):
            if len(self.rooms_included) > 0 and txt.split("@")[0] not in self.rooms_included:
                continue
            with open(os.path.join(self.root, "txt", txt), 'r') as f:
                lines = f.readlines()
                lines = [line.strip().split(' ') for line in lines]
                for line in lines:
                    overlap_ratio = float(line[2])
                    if overlap_ratio < self.overlap_dn or overlap_ratio > self.overlap_up:
                        continue # only acquire ply pairs in valid overlap ratio range
                    self.files.append(
                        (
                            line[0].split('.')[0]+'@'+line[1].split('.')[0].split('@')[1]+'@'+line[2],
                            os.path.join(self.root, "npz", line[0]),
                            os.path.join(self.root, "npz", line[1])
                        )
                    )
        
        if shuffle:
            np.random.shuffle(self.files)
        
        self.iterate_pos = -1
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        sample_name, frag1_path, frag2_path = self.files[idx]
        frag1 = utils.npz2npy(frag1_path)
        frag2 = utils.npz2npy(frag2_path)
        # add dummy uvw attributes
        frag1 = np.concatenate((frag1, np.zeros((len(frag1), 3))), axis=1)
        frag2 = np.concatenate((frag2, np.zeros((len(frag2), 3))), axis=1)

        T_gdth = np.eye(4)
        if self.augment:
            T_gdth = utils.build_random_transform(self.augdgre, self.augdist)
            frag2 = utils.apply_transformation(frag2, T_gdth)

        return frag1, frag2, T_gdth, sample_name

    def __next__(self):
        self.iterate_pos += 1
        if self.iterate_pos >= len(self.files):
            raise StopIteration
        return self[self.iterate_pos]

    def __iter__(self):
        return self

class KITTIOdometry(PairDataset):
    def __init__(
            self, 
            root: str, 
            shuffle: bool, 
            augment: bool, 
            augdgre: float, 
            augdist: float,
            args=None
        ) -> None:
        super().__init__(root, shuffle, augment, augdgre, augdist)

        self.voxel_size = args.prefilter_size
        self.filter_radius = args.filter_radius
        self.filter_mustnn = args.filter_mustnn

        self.step_size = args.step_size
        if self.step_size > 10:
            raise Exception("step size exceeds limitation 10")
        
        self.img_ids = [item[:-4] for item in sorted(os.listdir(f"{self.root}/image_0"))][::self.step_size]
        with open(f"{self.root}/poses.txt", 'r') as f:
            self.pos_lns = f.readlines()

        self.stereo_sgbm = cv2.StereoSGBM_create(
            0,          # minDisparity 最小可能差异值。通常情况下它是0
            96,         # numDisparities 最大差异减去最小差异。该值总是大于零。在当前的实现中，该参数必须可以被16整除
            9,          # BLOCKSIZE 匹配的块大小。它必须是> = 1的奇数。通常情况下，它应该在3..11的范围内
            # 该算法需要P2>P1。请参见stereo_match.cpp示例，其中显示了一些相当好的P1和P2值（分别为8 * number_of_image_channels * SADWindowSize * SADWindowSize和32 * number_of_image_channels * SADWindowSize * SADWindowSize）
            8 * 9 * 9,  # P1 控制视差平滑度的第一个参数
            32 * 9 * 9, # P2 第二个参数控制视差平滑度。值越大，差异越平滑。P1是相邻像素之间的视差变化加或减1的惩罚。P2是相邻像素之间的视差变化超过1的惩罚
            1, 
            63, 
            10, 
            100, 
            32
        )

        with open(f"{self.root}/calib.txt", 'r') as f:
                lines = f.readlines()
                # we only use P0 and P1, two gray image camera
                intrinsics_P0 = [float(str_val) for str_val in lines[0].rstrip().split(' ')[1:]] 
                intrinsics_P1 = [float(str_val) for str_val in lines[1].rstrip().split(' ')[1:]]
        self.fx =  intrinsics_P0[0]
        self.fy =  intrinsics_P0[5]
        self.cx =  intrinsics_P0[2]
        self.cy =  intrinsics_P0[6]
        self.bs = -intrinsics_P1[3] / self.fx

        self.iterate_pos = -1

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if idx == len(self.img_ids) - 1:
            idx -= 1 # in case the last frame is chosen
        frag1, T1 = self.png2npy_onthefly(idx)
        frag2, T2 = self.png2npy_onthefly(idx + 1)

        T_gdth = np.eye(4)
        if self.augment:
            T_gdth = utils.build_random_transform(self.augdgre, self.augdist)
            frag2 = utils.apply_transformation(frag2, T_gdth)
        
        sample_name = self.img_ids[idx] + "@" + self.img_idx[idx + 1]
        return frag1, frag2, T_gdth, sample_name

    def __next__(self):
        self.iterate_pos += 1
        if self.iterate_pos >= len(self.img_ids):
            raise StopIteration
        return self[self.iterate_pos]

    def __iter__(self):
        return self

    def png2npy_onthefly(self, idx):
        import open3d as o3d

        left_gray_img = cv2.imread(f"{self.root}/image_0/{self.img_ids[idx]}.png", 0) # 0-255 np.ndarray
        righ_gray_img = cv2.imread(f"{self.root}/image_1/{self.img_ids[idx]}.png", 0) # 0-255 np.ndarray
        disparity = self.stereo_sgbm.compute(
            left_gray_img,
            righ_gray_img
        ).astype(np.float32) / 16.0
        
        points_list = []
        ROWS, COLS = left_gray_img.shape
        for i in range(ROWS):
            for j in range(COLS):
                if disparity[i, j] < 10.0 or disparity[i, j] > 64.0:
                    continue
                # x, y, z in original space
                depth = self.fx * self.bs / disparity[i, j]
                points_list.append([
                    (j - self.cx) / self.fx * depth,
                    (i - self.cy) / self.fy * depth,
                    depth, 
                    left_gray_img[i, j],
                    left_gray_img[i, j],
                    left_gray_img[i, j]
                ])
        points_list = np.array(points_list)

        # one line in ground truth file
        # r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz

        # homogeneous 4x4 transformation matrix
        # r11 r12 r13 tx
        # r21 r22 r23 ty
        # r31 r32 r33 tz
        # 0   0   0   1
        T = np.array([float(str_val) for str_val in self.pos_lns[idx].strip().split(' ')], dtype=np.float32).reshape((3, 4))
        R = T[:3, :3].T
        t = T[:3,  3]
        points_list[:, :3] = points_list[:, :3] @ R + t
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = t

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_list[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(points_list[:, 3:])
        pcd.estimate_normals()

        points_list = np.concatenate([pcd.points, pcd.colors, pcd.normals], axis=1)
        points_list = utils.voxel_down_sample(points_list, self.voxel_size)
        points_list = utils.radius_outlier_filter(points_list, self.filter_radius, self.filter_mustnn)

        return points_list, T
