#!/bin/bash

SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

workspaceFolder=${SCRIPT_PATH}/..

python ${SCRIPT_PATH}/dump_fpfh_matches.py \
    --data_type ModelNet40Dense \
    --data_root /home/hm/fuguiduo/datasets/modelnet40/ply \
    --out_root ${workspaceFolder}/data/matches_modelnet40 \
    --classes "toilet night_stand radio sofa dresser" \
    --voxel_size 0.01 \
    --partition 0.00\
    --key_radius_factor 2.25 \
    --lambda1 2.75 \
    --lambda2 2.50 \
    --extracter_type FPFHFeatExtracter \
    --fpfh_radius_factor 1.75 \
    --fpfh_nn 50
