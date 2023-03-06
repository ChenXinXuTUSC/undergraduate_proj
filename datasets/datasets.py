import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tqdm import tqdm

import utils

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
            augdist: float
        ) -> None:
        super().__init__(root, shuffle, augment, augdgre, augdist)
        self.files = []
        mdirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
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
        self.iterate_pos += 1
        if self.iterate_pos >= len(self.files):
            raise StopIteration
        return self[self.iterate_pos]

    def __iter__(self):
        return self

class ThreeDMatchFCGF(PairDataset):
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
        # npzs = [os.path.join(root, "npz", file) for file in sorted(os.listdir(os.path.join(root, "npz")))]
        txts = [os.path.join(self.root, "txt", file) for file in sorted(os.listdir(os.path.join(self.root, "txt")))]
        for txt in tqdm(txts, total=len(txts), ncols=100, desc=self.__class__.__name__):
            with open(os.path.join(self.root, "txt", txt), 'r') as f:
                lines = f.readlines()
                lines = [line.rstrip().split(' ') for line in lines]
                for line in lines:
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

        if self.augment:
            frag2, rotmat, transd = utils.transform_augment(frag2, self.augdgre, self.augdist)
        
        T = np.zeros((4, 4))
        T[:3,:3] = rotmat
        T[:3 ,3] = transd
        T[3,3] = 1.0

        return frag1, frag2, T, sample_name

    def __next__(self):
        self.iterate_pos += 1
        if self.iterate_pos >= len(self.files):
            raise StopIteration
        return self[self.iterate_pos]

    def __iter__(self):
        return self
