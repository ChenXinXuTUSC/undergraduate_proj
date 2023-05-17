#!/bin/bash

CUDA_VISIBLE_DEVICES=0
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
workspaceFolder=${SCRIPT_PATH}/..


CUDA_VISIBLE_DEVICES=1
CUDA_LAUNCH_BLOCKING=1
python ${workspaceFolder}/statistic_count.py \
    --data_type ThreeDMatchFCGF \
    --data_root /home/hm/fuguiduo/datasets/3DMatch-FCGF \
    --out_root ${workspaceFolder}/results/statistic \
    --rooms "" \
    --voxel_size 0.05 \
    --overlap_dn 0.35 \
    --overlap_up 1.00 \
    \
    --key_radius_factor 2.25 \
    --lambda1 3.00 \
    --lambda2 3.00 \
    \
    --extracter_type FCGFFeatExtracter \
    --extracter_weight /home/hm/fuguiduo/code/DGR.mink/ResUNetBN2C-feat32-3dmatch-v0.05.pth \
    --fcgf_model ResUNetBN2C \
    \
    --mapper_conf ${workspaceFolder}/models/conf/mapper_3DMatch.yaml \
    --predicter_conf ${workspaceFolder}/models/conf/predicter_3DMatch.yaml \
    --positive_thresh 0.65 \
    \
    --recompute_norm \
    --salt_keypts 0.10
