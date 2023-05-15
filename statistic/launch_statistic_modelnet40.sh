#!/bin/bash

CUDA_VISIBLE_DEVICES=0
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
workspaceFolder=${SCRIPT_PATH}/..


CUDA_VISIBLE_DEVICES=1
CUDA_LAUNCH_BLOCKING=1
python ${workspaceFolder}/statistic_count.py \
    --data_type ModelNet40Dense \
    --data_root /home/hm/fuguiduo/datasets/modelnet40/ply \
    --out_root ${workspaceFolder}/results/statistic \
    --voxel_size 0.01 \
    --classes "" \
    --partition 0.00 \
    \
    --key_radius_factor 2.25 \
    --lambda1 2.50 \
    --lambda2 2.50 \
    \
    --extracter_type FPFHFeatExtracter \
    --fpfh_radius_factor 1.75 \
    --fpfh_nn 100 \
    \
    --mapper_conf ${workspaceFolder}/models/conf/mapper_ModelNet40.yaml \
    --predicter_conf ${workspaceFolder}/models/conf/predicter_ModelNet40.yaml \
    --positive_thresh 0.65 \
    \
    --recompute_norm \
    --salt_keypts 0.025 \
