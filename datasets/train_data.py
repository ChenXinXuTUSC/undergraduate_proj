import os
import torch
import numpy as np


class MatchingFCGF(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_root: str, 
        n_feat_per_sample: int, 
        postive_ratio: float=0.5
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.files = os.listdir(data_root)
        self.n_feat_per_sample = n_feat_per_sample
        self.positive_ratio = postive_ratio
        
        self.nP = int(self.n_feat_per_sample * self.positive_ratio) # num positive matches
        self.nN = int(self.n_feat_per_sample - self.nP)             # num negative matches
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        npz = np.load(f"{self.data_root}/{self.files[index]}")
        
        features = npz["features"]
        labels   = npz["labels"]
        
        labels_nP = labels.astype(np.int32).sum()
        labels_nN = len(labels) - labels_nP
        prepared_featt = []
        prepared_truth = []
        if labels_nP > 0:
            Pidx = np.random.choice(np.nonzero(labels.astype(np.int32))[0],  size=min(self.nP, labels_nP))
            prepared_featt.append(features[Pidx])
            prepared_truth.append(labels[Pidx])
        if labels_nN > 0:
            Nidx = np.random.choice(np.nonzero(~labels.astype(np.int32))[0], size=self.nN+max(self.nP-labels_nP, 0))
            prepared_featt.append(features[Nidx])
            prepared_truth.append(labels[Nidx])
        
        return (
            torch.from_numpy(np.concatenate(prepared_featt, axis=0)),
            torch.from_numpy(np.concatenate(prepared_truth, axis=0))
        )
