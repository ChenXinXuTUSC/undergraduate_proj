import numpy as np
from easydict import EasyDict as edict
from datasets import datasets
import utils
import config


if __name__ == "__main__":
    args = edict(vars(config.args))
    
    available_datasets = {attr_name: getattr(datasets, attr_name) for attr_name in dir(datasets) if callable(getattr(datasets, attr_name))}
    dataloader = available_datasets[args["data_type"]](
        root=args["data_root"],
        shuffle=True,
        augment=True,
        augdgre=60.0,
        augdist=2.0,
        args=args
    )

    for a, b, T, name in dataloader:
        utils.dump1frag(a, utils.make_ply_vtx_type(has_rgb=True, has_normal=True), out_name="origin.ply")
        a, *_ = utils.voxel_down_sample_gpt(a, args.ICP_radius, use_avg=True)
        utils.dump1frag(a, utils.make_ply_vtx_type(has_rgb=True, has_normal=True), out_name="dsampl.ply")
        break
