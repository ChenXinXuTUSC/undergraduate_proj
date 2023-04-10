import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True, help="path to dataset")
parser.add_argument("--data_type", type=str, required=True, help="ModelNet40Dense or 3DMatchFCGF or KITTI odometry")
parser.add_argument("--out_root", type=str, default="./results", help="dir to store the registration results")

# register model configurations
parser.add_argument("--voxel_size", type=float, required=True, help="voxel size for down sample and ICP registration")
parser.add_argument("--key_radius_factor", type=float, default=1.0, help="scale factor on voxel size for key point detection")
parser.add_argument("--extracter_type", type=str, default="FPFH", help="feature extracter model class name")
parser.add_argument("--feat_model", type=str, help="class name of feature extraction model")
parser.add_argument("--state_dict", type=str, help="path to the model state dict")

# dataset related configurations
# modelnet40
parser.add_argument("--classes", type=str, default="", help="model classes seperated by space, e.g., 'a b c d...'")
parser.add_argument("--partition", type=float, default=0.25, help="whether to partition point cloud into two parts or not")
# 3DMatch
parser.add_argument("--overlap_up", type=float, default=0.3, help="upper threshold of overlap ratio of the sample pair")
parser.add_argument("--overlap_dn", type=float, default=0.5, help="lower thresholf of overlap ratio of the sample pair")
parser.add_argument("--rooms", type=str, default="", help="room names seperated by space, e.g., 'a b c d...'")
# KITTI odometry
parser.add_argument("--step_size", type=int, help="interleave between two frames")
parser.add_argument("--prefilter_size", type=float, help="prefilter size for stereo vision point cloud fushion")
parser.add_argument("--filter_radius", type=float, help="radius to filter outliers")
parser.add_argument("--filter_mustnn", type=int,   help="must neighbours num to filter outliers")

# MinkowskiEngine test
args = parser.parse_args()
