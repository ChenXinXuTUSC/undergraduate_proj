import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True, help="path to dataset")
parser.add_argument("--data_type", type=str, required=True, help="ModelNet40Dense or 3DMatchFCGF or KITTI odometry")
parser.add_argument("--out_root", type=str, default="./results", help="dir to store the registration results")

# register model configurations
parser.add_argument("--voxel_size", type=float, required=True, help="voxel size for down sample and ICP registration")
# detector configuration
parser.add_argument("--key_radius_factor", type=float, default=1.0, help="scale factor on voxel size for key point detection")
parser.add_argument("--positive_thresh", type=float, default=0.5, help="whether to select a pair as postive according to its score")
parser.add_argument("--lambda1", type=float, default=2.75, help="ISS eigen value compare ratio 1")
parser.add_argument("--lambda2", type=float, default=2.65, help="ISS eigen value compare ratio 2")
parser.add_argument("--salt_keypts", action="store_true", help="whether to add salt-pepper keypoints")
# extractor configuration
parser.add_argument("--fpfh_radius_factor", type=float, default=1.0, help="scale factor on voxel size for feature extration")
parser.add_argument("--fpfh_nn", type=int, default=30, help="neighbour num used in each feature generation of fpfh")
parser.add_argument("--extracter_type", type=str, default="FPFHFeatExtracter", help="feature extracter model class name")
parser.add_argument("--extracter_weight", type=str, default="", help="path to the model state dict")
parser.add_argument("--fcgf_model", type=str, default="ResUNetBN2C", help="class name of feature extraction model")
# classifier configuration
parser.add_argument("--mapper_conf", type=str, default="", help="path to mapper conf yaml")
parser.add_argument("--predicter_conf", type=str, default="", help="path to predictor conf yaml")

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

# misc configuration
parser.add_argument("--recompute_norm", action="store_true", help="whether recompute the normal")

# MinkowskiEngine test
args = parser.parse_args()
