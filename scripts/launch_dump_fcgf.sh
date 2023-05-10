#!/bin/bash

SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

workspaceFolder=${SCRIPT_PATH}/..

python ${SCRIPT_PATH}/dump_fcgf_matches.py \
    --data_type ThreeDMatchFCGF \
    --data_root /home/hm/fuguiduo/datasets/3DMatch-FCGF \
    --out_root ${workspaceFolder}/data/matches_3DMatch \
    --overlap_dn 0.3 \
    --overlap_up 0.5 \
    --fcgf_model ResUNetBN2C \
    --extracter_weight /home/hm/fuguiduo/code/DGR.mink/ResUNetBN2C-feat32-3dmatch-v0.05.pth \
    --voxel_size 0.05 \
    --key_radius_factor 2.15 \
    --lambda1 2.75 \
    --lambda2 2.75 \
