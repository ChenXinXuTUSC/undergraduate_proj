import os
import numpy as np
import torch

from easydict import EasyDict as edict

import utils

from . import inlier_proposal

def snapshot(
        x: np.ndarray,
        y: np.ndarray,
        d: int,
        title: str="snapshot",
        out_dir: str=".",
        out_name: str="snapshot"
    ):
    main_axes = utils.principle_K_components(x, d)
    x = np.dot(x, main_axes)
    import matplotlib.pyplot as plt
    import matplotlib
    
    cmap = matplotlib.colormaps["plasma"]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=plt.figaspect(0.4))
    ax = fig.add_subplot(1, 1, 1)
    pidx = np.nonzero(y)[0]
    nidx = np.nonzero(~y)[0]
    if d == 2:
        pposes = (x[pidx, 0], x[pidx, 1])
        nposes = (x[nidx, 0], x[nidx, 1])
    elif d == 3:
        pposes = (x[pidx, 0], x[pidx, 1], x[pidx, 2])
        nposes = (x[nidx, 0], x[nidx, 1], x[nidx, 2])
    ax.scatter(*pposes, color=cmap(norm(y[pidx])), label=1)
    ax.scatter(*nposes, color=cmap(norm(y[nidx])), label=0)
    ax.title.set_text(title)
    ax.legend(loc="upper right")
    
    plt.savefig(f"{out_dir}/{out_name}.jpg")

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
        # matches filter
        mapper_conf: str,
        predictor_conf: str,
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
        
        self.extracter = None
        extracter_class = featextracter.load_extracter(extracter_type)
        if extracter_type == "FPFHFeatExtracter":
            self.extracter = extracter_class(
                voxel_size * feat_radius_factor,
                feat_neighbour_num
            )
        elif extracter_type == "FCGFFeatExtracter":
            self.extracter = extracter_class(
                model_type="ResUNetBN2C",
                state_dict_path=extracter_weights
            )
        
        self.ransac_params = edict({
            "max_workers": ransac_workers_num, "num_samples":ransac_samples_num,
            "max_corresponding_dist":voxel_size * ransac_corrdist_factor,
            "max_iter_num":ransac_iter_num,
            "max_valid_num":ransac_vald_num,
            "max_refine_num":ransac_rfne_num
        })
        self.checkr_params=edict({
            "max_corresponding_dist":voxel_size * checkr_corrdist_factor,
            "max_mnn_dist_ratio":checkr_mutldist_factor,
            "normal_angle_threshold":checkr_normdegr_thresh
        })

        # filter configuration
        self.mapper = None
        self.predictor = None
        self.use_filter = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if os.path.exists(mapper_conf):
            self.mapper = inlier_proposal.mapper.Mapper.conf_init(mapper_conf)
            self.mapper.to(self.device)
            self.mapper.eval()
        if os.path.exists(predictor_conf):
            self.predictor = inlier_proposal.predictor.Predictor.conf_init(predictor_conf)
            self.predictor.to(self.device)
            self.predictor.eval()
        if self.mapper is not None and self.predictor is not None:
            self.use_filter = True
        
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
        keyptsdict = utils.iss_detect(downsampled_coords, self.voxel_size * self.key_radius_factor)
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
        keyptsidx1,
        keyptsidx2,
        feats1: np.ndarray,
        feats2: np.ndarray,
        T_gdth: np.ndarray=None
    ):
        keycoords1 = downsampled_coords1[keyptsidx1]
        keycoords2 = downsampled_coords2[keyptsidx2]
        
        from utils import ransac
        keyfeats1 = feats1[keyptsidx1]
        keyfeats2 = feats2[keyptsidx2]
        # use feature descriptor of key points to compute matches
        matches = ransac.init_matches(keyfeats1.T, keyfeats2.T)
        totl_matches = np.array([keyptsidx1[matches[:,0]], keyptsidx2[matches[:,1]]]).T
        gdth_matches = None
        if T_gdth is not None:
            correct = utils.ground_truth_matches(matches, keycoords1, keycoords2, self.voxel_size * 2.5, T_gdth) # 上帝视角
            correct_valid_num = correct.astype(np.int32).sum()
            correct_total_num = correct.shape[0]
            utils.log_info(f"gdth/init: {correct_valid_num:d}/{correct_total_num:d}={correct_valid_num/correct_total_num:.3f}")
            gdth_matches = totl_matches[correct]
        
        if self.use_filter:
            matches, predicted_mask, manifold_coords = self.matches_filter(keyfeats1, keyfeats2, matches)
            snapshot(manifold_coords, correct,        d=2, out_name="gdth")
            snapshot(manifold_coords, predicted_mask, d=2, out_name="pred")
            correct_valid_num = np.logical_and(correct, predicted_mask).sum()
            correct_total_num = matches.shape[0]
            utils.log_dbug(f"gdth/pred: {correct_valid_num:d}/{correct_total_num:d}={correct_valid_num/correct_total_num:.3f}")
        
        coarse_registration = utils.ransac_match(
            keycoords1, keycoords2,
            keyfeats1, keyfeats2,
            ransac_params=self.ransac_params,
            checkr_params=self.checkr_params,
            matches=matches
        )
        
        return coarse_registration, totl_matches, gdth_matches
    
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
            downsampled_coords2,
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
            coords1 = copy.deepcopy(pcd1)[:,:3]
        if pcd2.shape[1] > 3:
            coords2 = copy.deepcopy(pcd2)[:,:3]
        
        # step1: voxel downsample
        downsampled_coords1, voxelized_coords1, idx_dse2vox1 = self.downsample(coords1)
        downsampled_coords2, voxelized_coords2, idx_dse2vox2 = self.downsample(coords2)
        
        # step2: detect iss key points
        keyptsdict1 = self.keypoints_detect(downsampled_coords1)
        keyptsdict2 = self.keypoints_detect(downsampled_coords2)
        
        # step3: compute feature descriptors for all points
        feats1 = self.extract_features(downsampled_coords1, voxelized_coords1)
        feats2 = self.extract_features(downsampled_coords2, voxelized_coords2)
        
        # step4: coarse registration
        coarse_registration, totl_matches, gdth_matches = self.coarse_registration(
            downsampled_coords1, downsampled_coords2,
            keyptsdict1["id"].values, keyptsdict2["id"].values,
            feats1, feats2,
            T_gdth
        )
        
        # step5: fine registration
        fine_registrartion = self.fine_registrartion(
            downsampled_coords1, downsampled_coords2,
            coarse_registration,
            10
        )
        
        return (
            fine_registrartion,
            pcd1[idx_dse2vox1], pcd2[idx_dse2vox2],
            keyptsdict1, keyptsdict2,
            totl_matches, gdth_matches
        )

    # other utilities
    def matches_filter(
        self,
        keyfeats1: np.ndarray,
        keyfeats2: np.ndarray,
        matches: np.ndarray,
    ):
        '''use contrastive model to filter matches
        
        params
        -
        * feats1: np.ndarray.
            Features in shape(num, dimensions).
        * feats2: np.ndarray.
            Features in shape(num, dimensions).
        * matches: np.ndarray.
            Corresponding matching features indices in shape(num, 2)
        
        return
        -
        * matches: np.ndarray.
            New filtered matches indices.
        '''
        
        with torch.no_grad():
            concat_feats = torch.from_numpy(
                np.concatenate([
                    keyfeats1[matches[:, 0]],
                    keyfeats2[matches[:, 1]]
                ], axis=1)
            )
            manifold_coords = self.mapper(concat_feats.unsqueeze(0).transpose(1,2).to(self.device))
            predicted_mask = (self.predictor(manifold_coords).transpose(1,2).squeeze().sigmoid().cpu().numpy()) > 0.75
        
        return matches[predicted_mask], predicted_mask, manifold_coords.reshape(-1, 3).cpu().numpy()
