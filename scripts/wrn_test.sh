#!/bin/bash

seeds=(0 1 2 3 4)
strategies=(random lc entropy sm)
dataset_dir="./data/bloodmnist_28"

for seed in "${seeds[@]}"; do
    for strategy in "${strategies[@]}"; do
        python test_wrn.py \
            --outdir=./wrn-runs/${strategy}/0000${seed}-run \
            --test_dir=./data/bloodmnist_28/test \
            --batch_size=256 \
            --cond=1 \
            --seed=1
    done
done