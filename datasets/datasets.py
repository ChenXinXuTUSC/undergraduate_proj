import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from plyfile import PlyData, PlyElement

import utils

class PairDataset:
    def __init__(
            self,
            root:str,
            shuffle: bool,
            augment: bool,
            augdgre: float,
            augdist: float
        ) -> None:
        self.root = root
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
            augdist: float
        ) -> None:
        super().__init__(root, shuffle, augment, augdgre, augdist)
        self.files = []
        mdirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        for mdir in mdirs:
            plys = sorted([ply for ply in os.listdir(os.path.join(root, mdir)) if ply.endswith(".ply")])
            for ply in plys:
                self.files.append((mdir, ply[:-4], os.path.abspath(os.path.join(root, mdir, ply))))
        
        if shuffle:
            self.files = self.files[np.random.choice(range(len(self.files)), replace=False)]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        _, sample_name, sample_path = self.files[idx]
        # # if sample is in txt format
        # with open(sample_path, 'r') as f:
        #     lines = f.readlines()
        #     lines = list(map(lambda li: [float(x) for x in li], [line.rstrip().split(',') for line in lines])) 
        # points = np.asarray(lines)

        points = np.asarray(PlyData.read(sample_path)["vertex"].data)
        # add dummy rgb attributes
        points = np.hstack(
            (
                points['x'].reshape((-1,1)),
                points['y'].reshape((-1,1)),
                points['z'].reshape((-1,1)),
                np.zeros((len(points), 1)),
                np.zeros((len(points), 1)),
                np.zeros((len(points), 1)),
                points["nx"].reshape((-1,1)),
                points["ny"].reshape((-1,1)),
                points["nz"].reshape((-1,1)),
            )
        )

        points_aug = points.copy()
        rotmat = np.identity(3)
        transd = np.array([0,0,0])

        if self.augment:
            points_aug, rotmat, transd = utils.transform_augment(points_aug, self.augdgre, self.augdist)
        
        T = np.zeros((4, 4))
        T[:3,:3] = rotmat
        T[:3 ,3] = transd
        T[3,3] = 1.0
        return points, points_aug, T, sample_name

    def __next__(self):
        for i in range(len(self.files)):
            return self.__getitem__(i)

    def __iter__(self):
        return self



