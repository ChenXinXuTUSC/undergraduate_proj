import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from datasets import datasets
import utils
import config


if __name__ == "__main__":
    args = vars(config.args)
    available_datasets = {attr_name: getattr(datasets, attr_name) for attr_name in dir(datasets) if callable(getattr(datasets, attr_name))}
    utils.log_dbug("hello world", available_datasets)
    dataloader = available_datasets[args["data_type"]](
        root=args["data_root"],
        shuffle=True,
        augment=True,
        augdgre=60.0,
        augdist=2.0
    )

    utils.log_dbug(f"{args['data_type']} dataset contains {len(dataloader)} samples")
    for a, b, T, name in dataloader:
        utils.log_dbug("sample name:", name)
    ply_line_type = np.dtype(
                [
                    ("x", "f4"), 
                    ("y", "f4"),
                    ("z", "f4"), 
                    ("red", "u1"), 
                    ("green", "u1"), 
                    ("blue", "u1"),
                    ("nx", "f4"),
                    ("ny", "f4"),
                    ("nz", "f4")
                ]
            )
    utils.fuse2frags(a, b, ply_line_type, out_name=name+".ply")
