import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from datasets import datasets
import utils
import config


if __name__ == "__main__":
    args = vars(config.args)
    utils.log_dbug(f"{args}")
    dataloader = datasets.ModelNet40Dense(
        root=args["data_root"],
        shuffle=False,
        augment=True,
        augdgre=30.0,
        augdist=2.0
    )

    utils.log_dbug(f"ModelNet40 dataset contains {len(dataloader)} samples")
    a, b, T, name = dataloader[0]
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
