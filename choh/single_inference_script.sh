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

threshold_array=(0 0.3 0.6 0.9 1.2 1.5)

for threshold in "${threshold_array[@]}"
do
    synthetic_nerf_command="CUDA_VISIBLE_DEVICES=${GPU} \
                            python ../experiment.py /external-volume/dataset/nerf_synthetic/${target_object} \
                            --workspace ./nerf_synthetic/nerf_synthetic_${target_object} \
                            -O \
                            --test \
                            --bound 1.0 \
                            --scale 0.8 \
                            --dt_gamma 0 \
                            --threshold ${threshold}"
    eval $synthetic_nerf_command
done

