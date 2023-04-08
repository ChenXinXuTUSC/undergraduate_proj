import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True, help="path to 3DMatach-FCGF dataset")
parser.add_argument("--data_type", type=str, required=True, help="ModelNet40Dense or 3DMatchFCGF")
parser.add_argument("--out_root", type=str, default="./results", help="dir to store the registration results")
parser.add_argument("--ICP_radius", type=float, required=True, help="radius for down sample and ICP registration")
# dataset related options

# modelnet40
parser.add_argument("--classes", type=str, default="", help="model classes seperated by space, e.g., 'a b c d...'")
parser.add_argument("--partition", type=float, default=0.25, help="whether to partition point cloud into two parts or not")
# 3DMatch
parser.add_argument("--overlap_up", type=float, default=0.3, help="upper threshold of overlap ratio of the sample pair")
parser.add_argument("--overlap_dn", type=float, default=0.5, help="lower thresholf of overlap ratio of the sample pair")
# KITTI odometry
parser.add_argument("--step_size", type=int, help="interleave between two frames")
parser.add_argument("--voxel_size", type=float, help="voxel size for down sampling")
parser.add_argument("--filter_radius", type=float, help="radius to filter outliers")
parser.add_argument("--filter_mustnn", type=int,   help="must neighbours num to filter outliers")

# MinkowskiEngine test
parser.add_argument("--feat_model", type=str, help="class name of feature extraction model")
parser.add_argument("--state_dict", type=str, help="path to the model state dict")
args = parser.parse_args()
