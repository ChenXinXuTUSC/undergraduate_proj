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
        # keypoint detector
        detecter_conf,
        # feature extracter
        extracter_conf,
        # inlier proposal
        mapper_conf: str,
        predicter_conf: str,
        # optimization
        ransac_conf,
        checkr_conf,
        misc=None
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
        self.misc = misc # other configuration
        
        from . import featextracter
        self.voxel_size = voxel_size
        
        self.detecter_conf = detecter_conf
        
        self.extracter = featextracter.load_extracter(extracter_conf)
        
        self.ransac_conf = ransac_conf
        self.checkr_conf = checkr_conf

        # filter configuration
        self.mapper = None
        self.predicter = None
        self.use_filter = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if os.path.exists(mapper_conf):
            self.mapper = inlier_proposal.mapper.Mapper.conf_init(mapper_conf)
            self.mapper.to(self.device)
            self.mapper.eval()
        if os.path.exists(predicter_conf):
            self.predicter = inlier_proposal.predicter.Predicter.conf_init(predicter_conf)
            self.predicter.to(self.device)
            self.predicter.eval()
        if self.mapper is not None and self.predicter is not None:
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
    def detect_keypoints(self, downsampled_coords: np.ndarray, add_salt:float=0.0):
        keyptsdict = utils.iss_detect(
            downsampled_coords,
            self.detecter_conf.key_radius,
            self.detecter_conf.lambda1,
            self.detecter_conf.lambda2
        )
        
        if add_salt > 1e-5:
            fullset = set(list(range(len(downsampled_coords))))
            subbset = set(list(keyptsdict["id"].values))
            rndptsidx = np.random.choice(list(fullset - subbset), size=int(len(downsampled_coords) * add_salt), replace=False)
            keyptsidx = np.concatenate([keyptsdict["id"].values, rndptsidx])
        else:
            rndptsidx = None
            keyptsidx = keyptsdict["id"].values
        
        return (
            keyptsidx, rndptsidx,
            keyptsdict["eigval1"].values,
            keyptsdict["eigval2"].values,
            keyptsdict["eigval3"].values
        ) 
    
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
        T_gdth: np.ndarray=None,
        matches: np.ndarray=None
    ):
        keycoords1 = downsampled_coords1[keyptsidx1]
        keycoords2 = downsampled_coords2[keyptsidx2]
        keyfeats1 = feats1[keyptsidx1]
        keyfeats2 = feats2[keyptsidx2]
        
        if matches is None:
            # use feature descriptor of key points to compute matches
            matches = utils.ransac.init_matches(keyfeats1.T, keyfeats2.T)
        
        totl_matches = np.array([keyptsidx1[matches[:,0]], keyptsidx2[matches[:,1]]]).T
        gdth_matches = None
        if T_gdth is not None:
            correct = utils.ground_truth_matches(matches, keycoords1, keycoords2, self.voxel_size * 1.50, T_gdth)
            correct_valid_num = correct.astype(np.int32).sum()
            correct_total_num = correct.shape[0]
            utils.log_info(f"gdth/init: {correct_valid_num:4d}/{correct_total_num:4d}={correct_valid_num/correct_total_num:.3f}")
            gdth_matches = totl_matches[correct]
        
        filtered_ratio = None
        if self.use_filter:
            matches, predicted_mask, manifold_coords = self.matches_filter(keyfeats1, keyfeats2, matches)
            plane_coords = np.reshape(manifold_coords, (-1, self.mapper.out_channels))
            # snapshot(plane_coords, correct,        d=2, out_name="gdth")
            # snapshot(plane_coords, predicted_mask, d=2, out_name="pred")
            num_valid_matches = np.logical_and(correct, predicted_mask).sum()
            num_total_matches = matches.shape[0]
            filtered_ratio = num_valid_matches / num_total_matches
            utils.log_dbug(f"gdth/pred: {num_valid_matches:4d}/{num_total_matches:4d}={filtered_ratio:.3f}")
            if num_valid_matches / num_total_matches < 0.1 or num_valid_matches < 3:
                return None, totl_matches, gdth_matches, filtered_ratio
        
        coarse_registration = utils.ransac_match(
            keycoords1, keycoords2,
            keyfeats1, keyfeats2,
            ransac_conf=self.ransac_conf,
            checkr_conf=self.checkr_conf,
            matches=matches
        )
        
        return coarse_registration, totl_matches, gdth_matches, filtered_ratio
    
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
        * pcd1(np.ndarray): Points in shape(n, feature_dimensions).
        * pcd2(np.ndarray): Points in shape(n, feature_dimensions).
        
        return
        -
        * registration(open3d.registration.RegistrationResult): Pre
            dicted registration result containing the T and corr.
        * downsampled_coords1(np.ndarray): Coordinates1 after voxel
            downsampled.
        * downsampled_coords2(np.ndarray): Coordinates2 after voxel
            downsampled.
        * keyptsidx1(np.ndarray): Index of key points in downsample
            coords1.
        * keyptsidx2(np.ndarray): Index of key points in downsample
            coords2.
        * totl_matches(np.ndarray): Totl matching pair  with  shape
            (n, 2).
        * vald_matches(np.ndarray): Valid matching pair with  shape
            (n, 2).
        * misc(EasyDict.easydict): Other things you want to  return
            but not that important compared to others.
        '''
        # misc return values
        miscret = dict()
        
        
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
        keyptsidx1, rndptsidx1, *_ = self.detect_keypoints(downsampled_coords1, self.misc.salt_keypts)
        keyptsidx2, rndptsidx2, *_ = self.detect_keypoints(downsampled_coords2, self.misc.salt_keypts)
        miscret["rndptsidx1"] = rndptsidx1
        miscret["rndptsidx2"] = rndptsidx2
        
        # step3: compute feature descriptors for all points
        feats1 = self.extract_features(pcd1[idx_dse2vox1], voxelized_coords1)
        feats2 = self.extract_features(pcd2[idx_dse2vox2], voxelized_coords2)
        
        # step4: coarse registration
        coarse_registration, totl_matches, gdth_matches, filtered_ratio = self.coarse_registration(
            downsampled_coords1, downsampled_coords2,
            keyptsidx1, keyptsidx2,
            feats1, feats2,
            T_gdth
        )
        miscret["filtered_ratio"] = filtered_ratio
        
        if coarse_registration is None:
            return (
                None,
                pcd1[idx_dse2vox1], pcd2[idx_dse2vox2],
                keyptsidx1, keyptsidx2,
                totl_matches, gdth_matches,
                edict(miscret)
            )
        # utils.log_dbug(f"coarse corresponding pairs: {len(coarse_registration.correspondence_set)}")
        
        # step5: fine registration
        fine_registrartion = self.fine_registrartion(
            downsampled_coords1, downsampled_coords2,
            coarse_registration,
            50
        )
        
        return (
            fine_registrartion,
            pcd1[idx_dse2vox1], pcd2[idx_dse2vox2], # retain other features, don't use downsampled_coords
            keyptsidx1, keyptsidx2,
            totl_matches, gdth_matches,
            edict(miscret)
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
            manifold_coords = self.mapper(concat_feats.unsqueeze(0).transpose(1,2).to(self.device).float())
            predicted_mask = (self.predicter(manifold_coords).transpose(1,2).squeeze().sigmoid().cpu().numpy()) > self.misc.positive_thresh
        
        return matches[predicted_mask], predicted_mask, manifold_coords.cpu().numpy()
