#!/bin/bash

SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

workspaceFolder=${SCRIPT_PATH}/..

python ${workspaceFolder}/dump_fcgf_matches.py \
    --data_type ThreeDMatchFCGF \
    --data_root /home/hm/fuguiduo/datasets/3DMatch-FCGF \
    --rooms 7-scenes-redkitchen \
    --out_root ${workspaceFolder}/data \
    --voxel_size 0.05 \
    --overlap_dn 0.3 \
    --overlap_up 0.5 \
    --feat_model ResUNetBN2C \
    --state_dict /home/hm/fuguiduo/code/DGR.mink/ResUNetBN2C-feat32-3dmatch-v0.05.pth
