#!/bin/bash

CUDA_VISIBLE_DEVICES=0
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
workspaceFolder=${SCRIPT_PATH}/..

# python ${SCRIPT_PATH}/noise_ratio_ModelNet40.py \
#     --data_type ModelNet40Dense \
#     --data_root /home/hm/fuguiduo/datasets/modelnet40/ply \
#     --out_root ${SCRIPT_PATH} \
#     --classes "radio monitor lamp vase bed" \
#     --voxel_size 0.01 \
#     --partition 0.75\
#     --key_radius_factor 2.25 \
#     --lambda1 2.25 \
#     --lambda2 2.25 \
#     --extracter_type FPFHFeatExtracter \
#     --fpfh_radius_factor 1.75 \
#     --fpfh_nn 100 \
#     --salt_keypts 0.00 \
#     --recompute_norm \

python ${SCRIPT_PATH}/noise_ratio_3DMatch.py \
    --data_type ThreeDMatchFCGF \
    --data_root /home/hm/fuguiduo/datasets/3DMatch-FCGF \
    --out_root ${SCRIPT_PATH} \
    --rooms "analysis-by-synthesis-apt1 bundlefusion-office sun3d-harvard sun3d-mit 7-scenes-redkitchen" \
    --overlap_dn 0.45 \
    --overlap_up 0.50 \
    --fcgf_model ResUNetBN2C \
    --extracter_weight /home/hm/fuguiduo/code/DGR.mink/ResUNetBN2C-feat32-3dmatch-v0.05.pth \
    --voxel_size 0.05 \
    --key_radius_factor 2.15 \
    --lambda1 2.75 \
    --lambda2 2.75 \
    --recompute_norm \
    --salt_keypts 0.00
