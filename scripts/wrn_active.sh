#!/bin/bash

seeds=(1 2 3 4 5)
strategies=(random lc entropy sm)
dataset_dir="./data/bloodmnist_224"

for seed in "${seeds[@]}"; do
    for strategy in "${strategies[@]}"; do
        accelerate launch train_wrn.py \
            --outdir=wrn-runs/${strategy} \
            --train_dir=${dataset_dir}/train \
            --val_dir=${dataset_dir}/val \
            --test_dir=${dataset_dir}/test \
            --batch_size=128 \
            --cond=1 \
            --num_epochs=50 \
            --accum_steps=1 \
            --decay_epochs=60,120,160 \
            --decay_rate=0.2 \
            --depth=28 \
            --width_factor=10 \
            --dropout_rate=0.0 \
            --lr=0.0001 \
            --seed=$seed \
            --eval_interval=5 \
            --exp_type=active \
            --num_samples=0.1 \
            --strategy=$strategy \
            --calibrate=0 \
            --train_on_latents=1
    done
done
