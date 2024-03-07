#!/bin/bash

# Initial Settings
GPU=0

# synthetic_nerf
# target_object="ship"
# target_object="mic"
# target_object="materials"
# target_object="lego"
# target_object="hotdog"
# target_object="ficus"
target_object="drums"
# target_object="chair"

synthetic_nerf_command="CUDA_VISIBLE_DEVICES=${GPU} \
                        python ../main_nerf.py /external-volume/dataset/nerf_synthetic/${target_object} \
                        --workspace ./nerf_synthetic/nerf_synthetic_${target_object} \
                        --iters 40000 \
                        -O \
                        --bound 1.0 \
                        --scale 0.8 \
                        --dt_gamma 0"

eval $synthetic_nerf_command