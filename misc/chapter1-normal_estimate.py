import open3d as o3d
import numpy as np

from datasets import datasets
import config
import utils

from easydict import EasyDict as edict

def pca_analysis(points:np.ndarray, correlation=False, sort=True):
    center = points.mean(axis=0)
    decentralized_points = points - center
    # compute data covariance matrix
    cov = np.matmul(decentralized_points.T, decentralized_points)
    # for 3d data, it has at most 3 bases
    # PCA和SVD的本质计算都在特征值分解这一步
    eigvecs, eigvals = np.linalg.eig(cov)

if __name__ == "__main__":
    args = edict(vars(config.args))

    available_datasets = {attr_name: getattr(datasets, attr_name) for attr_name in dir(datasets) if callable(getattr(datasets, attr_name))}
    dataloader = available_datasets[args.data_type](
        root=args.data_root,
        shuffle=True,
        augment=True,
        augdgre=30.0,
        augdist=4.0,
        args=args
    )

    for points1, points2, T_gdth, sample_name in dataloader:
        utils.log_info(sample_name)
        points1_o3d = utils.npy2o3d(points1)
        points1_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.ICP_radius*1.25, max_nn=30))
        points1_o3d.paint_uniform_color([0.0, 0.0, 0.0])
        points1 = utils.o3d2npy(points1_o3d)
        points1[:,3:6] = (points1[:,6:9] + 1.0) / 2.0 * 255.0

        # compute local covariance matrix for points2
        # to estimate normals


        utils.dump1frag(points1, utils.ply_vertex_type, out_dir="./samples/pca_analysis_visual", out_name="output.ply")
        break
