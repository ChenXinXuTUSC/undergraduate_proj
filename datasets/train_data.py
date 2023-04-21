import os
import torch
import numpy as np

import concurrent

import utils

class MatchingFeats(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_root: str, 
        num_matches_per_sample: int, 
        postive_ratio: float=0.5,
        filter_strs: list=None
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.num_matches_per_sample = num_matches_per_sample
        self.positive_ratio = postive_ratio
        
        self.nP = int(self.num_matches_per_sample * self.positive_ratio) # num positive matches
        self.nN = self.num_matches_per_sample - self.nP                  # num negative matches
        
        self.files = []
        # do the filtration
        if len(filter_strs) > 0:
            for file in os.listdir(data_root):
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(filter_strs)) as executor:
                    futures = [executor.submit(lambda x: x in file, filter_str) for filter_str in filter_strs]
                    for future in concurrent.futures.as_completed(futures):
                        if future.result():
                            self.files.append(file)
                            break
        else:
            self.files = os.listdir(data_root)
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        npz = np.load(f"{self.data_root}/{self.files[index]}")
        
        features = npz["features"]
        labels   = npz["labels"]
        
        nP_wanted = self.nP
        nN_wanted = self.nN
        nP_available = labels.astype(np.int32).sum()
        nN_available = len(labels) - nP_available
        prepared_featt = []
        prepared_truth = []

        try:
            Pidx = np.random.choice(np.nonzero(labels.astype(np.int32))[0],  size=min(nP_wanted, nP_available), replace=False)
            prepared_featt.append(features[Pidx])
            prepared_truth.append(labels[Pidx])
        except ValueError:
            Ridx = np.random.choice(len(features), size=nP_wanted, replace=True)
            prepared_featt.append(features[Ridx])
            prepared_truth.append(labels[Ridx])
        try:
            Nidx = np.random.choice(np.nonzero((~labels).astype(np.int32))[0], size=nN_wanted + max(nP_wanted - nP_available, 0), replace=True)
            prepared_featt.append(features[Nidx])
            prepared_truth.append(labels[Nidx])
        except ValueError:
            Ridx = np.random.choice(len(features), size=nN_wanted + max(nP_wanted - nP_available, 0), replace=True)
            prepared_featt.append(features[Ridx])
            prepared_truth.append(labels[Ridx])
        
        # # if sample matches are not enough
        # if len(prepared_featt[0]) + len(prepared_featt[1]) < self.num_matches_per_sample:
        #     filldex = np.random.choice(len(features), size=self.num_matches_per_sample - len(prepared_featt), replace=True)
        #     prepared_featt.append(features[filldex])
        #     prepared_truth.append(labels[filldex])
        
        return (
            torch.from_numpy(np.concatenate(prepared_featt, axis=0)),
            torch.from_numpy(np.concatenate(prepared_truth, axis=0)),
        )
