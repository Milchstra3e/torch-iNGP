export PYTHONPATH=${PWD}/iNGP/

python ./single_object_nerf.py /external-volume/dataset/nerf_synthetic/lego     \
                                --workspace data/lego                           \
                                -O                                              \
                                --bound 1.0                                     \
                                --scale 0.8                                     \
                                --dt_gamma 0                                    \
                                --batch_size 1                                  \
                                --group_grid_cnt 2                              \
                                --run_type multiple_cuda                        \
                                # --debug                                         \