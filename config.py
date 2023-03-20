import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True, help="path to 3DMatach-FCGF dataset")
parser.add_argument("--data_type", type=str, required=True, help="ModelNet40Dense or 3DMatchFCGF")
parser.add_argument("--ICP_radius", type=float, required=True, help="radius to down sample and ICP registration")
# dataset related options

# modelnet40
parser.add_argument("--classes", type=str, help="model classes seperated by space, e.g., 'a b c d...'")
# 3DMatch
parser.add_argument("--overlap_up", type=float, help="upper threshold of overlap ratio of the sample pair")
parser.add_argument("--overlap_dn", type=float, help="lower thresholf of overlap ratio of the sample pair")

# MinkowskiEngine test
parser.add_argument("--feat_model", type=str, help="class name of feature extraction model")
parser.add_argument("--state_dict", type=str, help="path to the model state dict")
args = parser.parse_args()
