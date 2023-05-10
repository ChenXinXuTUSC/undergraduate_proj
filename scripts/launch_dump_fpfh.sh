#!/bin/bash

SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

workspaceFolder=${SCRIPT_PATH}/..

python ${SCRIPT_PATH}/dump_fpfh_matches.py \
    --data_type ModelNet40Dense \
    --data_root /home/hm/fuguiduo/datasets/modelnet40/ply \
    --out_root ${workspaceFolder}/data/matches_ModelNet40 \
    --classes "airplane guitar" \
    --voxel_size 0.01 \
    --partition 0.00\
    --key_radius_factor 2.00 \
    --lambda1 2.00 \
    --lambda2 2.00 \
    --extracter_type FPFHFeatExtracter \
    --fpfh_radius_factor 1.75 \
    --fpfh_nn 100 \
    --recompute_norm
