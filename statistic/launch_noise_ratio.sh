#!/bin/bash

SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

workspaceFolder=${SCRIPT_PATH}/..

python ${SCRIPT_PATH}/noise_ratio.py \
    --data_type ModelNet40Dense \
    --data_root /home/hm/fuguiduo/datasets/modelnet40/ply \
    --out_root ${SCRIPT_PATH} \
    --classes "radio monitor lamp vase bed" \
    --voxel_size 0.01 \
    --partition 0.75\
    --key_radius_factor 2.25 \
    --lambda1 2.25 \
    --lambda2 2.25 \
    --extracter_type FPFHFeatExtracter \
    --fpfh_radius_factor 1.75 \
    --fpfh_nn 100 \
    --salt_keypts 0.00 \
    --recompute_norm \
