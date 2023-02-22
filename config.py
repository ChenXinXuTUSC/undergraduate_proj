import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, help="path to 3DMatach-FCGF dataset")
parser.add_argument("--ICP_radius", type=float, help="radius to down sample and ICP registration")
args = parser.parse_args()
