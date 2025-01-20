#!/bin/bash

seeds=(0)
strategies=(lc)
dataset_dir="./data/bloodmnist_224"

for seed in "${seeds[@]}"; do
    for strategy in "${strategies[@]}"; do
        python test_wrn.py \
            --model_type=egc \
            --outdir=./egc-runs/${strategy}/0000${seed}-run \
            --test_dir=${dataset_dir}/test \
            --train_on_latents=1
    done
done