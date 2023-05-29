#!/bin/bash

SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
workspaceFolder=${SCRIPT_PATH}/..


CUDA_VISIBLE_DEVICES=0
CUDA_LAUNCH_BLOCKING=1

python ${workspaceFolder}/o3d_testbench.py \
    --data_type ModelNet40Dense \
    --data_root /home/hm/fuguiduo/datasets/modelnet40/ply \
    --classes "airplane car guitar flower_pot table chair person glass_box" \
    --out_root ${workspaceFolder}/results/o3dtest \
    --voxel_size 0.01 \
    --partition 0.10 \
    \
    --key_radius_factor 1.5 \
    --fpfh_radius_factor 1.75 \
    --fpfh_nn 100
