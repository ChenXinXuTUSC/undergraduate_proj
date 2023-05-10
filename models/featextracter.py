import torch
import numpy as np
import open3d as o3d

import utils

class FPFHFeatExtracter:
    def __init__(
        self, 
        feat_radius: float,
        max_neighbournum: int=50
    ) -> None:
        self.radius = feat_radius
        self.max_nn = max_neighbournum
    
    @classmethod
    def conf_init(cls, conf):
        return cls(conf.feat_radius, conf.feat_neighbour_num)
    
    def __call__(self, downsampled_coords: np.ndarray, voxelized_coords: np.ndarray):
        coords_o3d = utils.npy2o3d(downsampled_coords)
        
        # please provide the normal before entering this function
        # do not recompute normal on downsampled points!
        # compute all points' fpfh
        fpfhs = o3d.pipelines.registration.compute_fpfh_feature(
            coords_o3d,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.radius,
                max_nn=self.max_nn
            )
        ).data.T
        
        return fpfhs

class FCGFFeatExtracter:
    def __init__(
        self,
        fcgf_model: str,
        path_to_weight: str
    ) -> None:
        from . import fcgf
        
        pth_data = torch.load(path_to_weight)
        self.model_configs = pth_data["config"]
        self.model_weights = pth_data["state_dict"]
        
        self.fcgf_model = fcgf.load_model(fcgf_model)(
            1,
            self.model_configs["model_n_out"],
            bn_momentum=self.model_configs["bn_momentum"],
            conv1_kernel_size=self.model_configs["conv1_kernel_size"],
            normalize_feature=self.model_configs["normalize_feature"]
        )
        self.fcgf_model.load_state_dict(self.model_weights)
        self.fcgf_model.eval()
        self.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fcgf_model.to(self.model_device)
        
    @classmethod
    def conf_init(cls, conf):
        return cls(conf.fcgf_model, conf.extracter_weight)
    
    def __call__(self, downsampled_coords: np.ndarray, voxelized_coords: np.ndarray):
        import MinkowskiEngine as ME
        fcgfs = self.fcgf_model(
            ME.SparseTensor(
                coordinates=ME.utils.batched_coordinates([voxelized_coords]).to(self.model_device), 
                features=torch.ones(len(voxelized_coords), 1).to(self.model_device)
            )
        ).F.detach().cpu().numpy()
        return fcgfs

ALL_FEATEXTRACTER = [FPFHFeatExtracter, FCGFFeatExtracter]
extracter_dict = {m.__name__:m for m in ALL_FEATEXTRACTER}
def load_extracter(extracter_conf):
    extracter_type = extracter_conf["extracter_type"]
    if extracter_type not in extracter_dict:
        raise Exception("feature extracter not recognized")
    return extracter_dict[extracter_type].conf_init(extracter_conf)
