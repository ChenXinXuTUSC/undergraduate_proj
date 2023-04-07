import numpy as np
import utils

import collections
#RANSAC configuration:
RANSACCONF = collections.namedtuple(
    "RANSACCONF",
    [
        "max_workers",
        "num_samples",
        "max_corresponding_dist", 'max_iter_num', 'max_valid_num', 'max_refine_num'
    ]
)
# fast pruning algorithm configuration:
CHECKRCONF = collections.namedtuple(
    "CHECKRCONF",
    [
        "max_corresponding_dist",
        "max_mnn_dist_ratio", 
        "normal_angle_threshold"
    ]
)

class RansacRegister:
    '''
    A ransac registor using ISS as key points detector
    
    '''
    def __init__(
        self,
        voxel_size: float,
        # ISS key point detector
        key_radius_factor: float,
        # FCGF and FPFH extracter
        extracter_type: str,
        extracter_weights: str,
        feat_radius_factor: float,
        feat_neighbour_num: int,
        # ransac registration
        ransac_workers_num: int,
        ransac_samples_num: int,
        ransac_corrdist_factor: float,
        ransac_iter_num: int,
        ransac_vald_num: int,
        ransac_rfne_num: int,
        checkr_corrdist_factor: float,
        checkr_mutldist_factor: float,
        checkr_normdegr_thresh: float
    ) -> None:
        '''
        param
        -
        * voxel_size: float.
            Voxel size used to down sample the point cloud.
        * extractor_type: str.
            Name of the feature extractor model,  could  be
            choose from [FPFH | FCGF], FPFH is a local  ex-
            tractor while FCGF is a global extractor.
        * extractor_weights: str
            Path to the weights of extractor model, if FCGF
            model is used, this must be provided.
        * ransac_workers_num: int.
            Multi-processing to find the inital ransac  tr-
            ansformation T[R, t].
        * ransac_corrdist_factor: float.
            Related to voxel size, used to determined if  a
            pair of point is corresponding point after  tr-
            ansformation [R,t], true if Euclidean  distance
            is less than this value, which will  influcence
            registration evaluation
        * ransac_iter_num: int.
            Maximum time of finding other T after initial T
            is found.
        * ransac_vald_num: int.
            Maximum time of finding  other  valid  T  after 
            inital T is found.
        * ransac_rfne_num: int.
            Maximum of time of ICP refining.
        * T_gdth: np.ndarray.
            Ground truth trasformation T[R,t]
        '''
        from . import featextracter
        self.voxel_size = voxel_size
        
        self.key_radius_factor = key_radius_factor
        
        extracter_class = featextracter.load_extracter(extracter_type)
        if extracter_type == "FPFH":
            self.extracter = extracter_class(
                voxel_size * feat_radius_factor,
                feat_neighbour_num
            )
        elif extracter_type == "FCGF":
            self.extracter = extracter_class(
                model_type="ResUNetBN2C",
                state_dict_path=extracter_weights
            )
        
        self.ransac_params = RANSACCONF(
            max_workers=ransac_workers_num, num_samples=ransac_samples_num,
            max_corresponding_dist=voxel_size * ransac_corrdist_factor,
            max_iter_num=ransac_iter_num,
            max_valid_num=ransac_vald_num,
            max_refine_num=ransac_rfne_num
        )
        self.checkr_params=CHECKRCONF(
            max_corresponding_dist=voxel_size * checkr_corrdist_factor,
            max_mnn_dist_ratio=checkr_mutldist_factor,
            normal_angle_threshold=checkr_normdegr_thresh
        )
    
    # step1: voxel downsample
    def downsample(self, coords: np.ndarray):
        downsampled_coords, voxelized_coords, idx_dse2vox = utils.voxel_down_sample_gpt(
            points=coords,
            voxel_size=self.voxel_size,
            use_avg=False
        )
        return downsampled_coords, voxelized_coords, idx_dse2vox
    
    # step2: detect keypoints
    def keypoints_detect(self, downsampled_coords: np.ndarray):
        keyptsdict = utils.iss_detect(downsampled_coords, self.voxel_size * self.ISS_radius_factor)
        return keyptsdict
    
    # step3: extract feature descriptors of all points
    def extract_features(self, downsampled_coords: np.ndarray, voxelized_coords: np.ndarray):
        feats = self.extracter(downsampled_coords, voxelized_coords)
        return feats
    
    # step4: coarse ransac registration
    def coarse_registration(
        self,
        downsampled_coords1: np.ndarray,
        downsampled_coords2: np.ndarray,
        keyptsdict1,
        keyptsdict2,
        feats1: np.ndarray,
        feats2: np.ndarray,
        T_gdth: np.ndarray=None
    ):
        from utils import ransac
        keyfeats1 = feats1[keyptsdict1["id"].values].T
        keyfeats2 = feats2[keyptsdict2["id"].values].T
        # use feature descriptor of key points to compute matches
        matches = ransac.init_matches(keyfeats1, keyfeats2)
        
        keycoords1 = downsampled_coords1[keyptsdict1["id"].values]
        keycoords2 = downsampled_coords2[keyptsdict2["id"].values]
        if T_gdth is not None:
            correct = utils.ground_truth_matches(matches, keycoords1, keycoords2, self.voxel_size * 2.5, T_gdth) # 上帝视角
            correct_valid_num = correct.astype(np.int32).sum()
            correct_total_num = correct.shape[0]
            utils.log_info(f"gdth/init: {correct_valid_num:.2f}/{correct_total_num:.2f}={correct_valid_num/correct_total_num:.2f}")
        
        coarse_registration = utils.ransac_match(
            keycoords1, keycoords2,
            keyfeats1, keyfeats2,
            ransac_params=self.ransac_params,
            checkr_params=self.checkr_params
        )
        
        return coarse_registration
    
    # step5: fine ransac registration
    def fine_registrartion(
        self,
        downsampled_coords1: np.ndarray,
        downsampled_coords2: np.ndarray,
        coarse_registration,
        num_finetune_steps
    ):
        from utils import icp
        import open3d as o3d
        return icp.ICP_exact_match(
            downsampled_coords1,
            downsampled_coords1,
            o3d.geometry.KDTreeFlann(utils.npy2o3d(downsampled_coords2)),
            coarse_registration.transformation,
            self.voxel_size,
            num_finetune_steps
        )

    
    def register(self, pcd1: np.ndarray, pcd2: np.ndarray, T_gdth: np.ndarray=None):
        '''predict the original transformation T[R,t]
        
        params
        -
        * pcd1: np.ndarray.
            points in shape(n, feature_dimensions).
        * pcd2: np.ndarray.
            points in shape(n, feature_dimensions).
        
        return
        -
        * T: np.ndarray
            Predicted transformation T in shape(4,4).
        '''
        import copy
        # we don't need features other than coordinates
        if pcd1.shape[1] > 3:
            pcd1 = copy.deepcopy(pcd1)[:,:3]
        if pcd2.shape[1] > 3:
            pcd2 = copy.deepcopy(pcd2)[:,:3]
        
        # step1: voxel downsample
        downsampled_coords1, voxelized_coords1, idx_dse2vox1 = self.downsample(pcd1)
        downsampled_coords2, voxelized_coords2, idx_dse2vox2 = self.downsample(pcd2)
        
        # step2: detect iss key points
        keyptsdict1 = self.keypoints_detect(downsampled_coords1)
        keyptsdict2 = self.keypoints_detect(downsampled_coords2)
        
        # step3: compute feature descriptors for all points
        feats1 = self.extract_features(downsampled_coords1, voxelized_coords1)
        feats2 = self.extract_features(downsampled_coords2, voxelized_coords2)
        
        # step4: coarse registration
        coarse_registration = self.coarse_registration(
            downsampled_coords1, downsampled_coords2,
            keyptsdict1, keyptsdict2,
            feats1, feats2,
            T_gdth
        )
        
        # step5: fine registration
        fine_registrartion = self.fine_registrartion(
            downsampled_coords1, downsampled_coords2,
            coarse_registration,
            100
        )
        
        return fine_registrartion
