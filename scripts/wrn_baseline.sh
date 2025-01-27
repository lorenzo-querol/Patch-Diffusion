#!/bin/bash

seeds=(1 2 3 4 5)
datasets=("bloodmnist_256" "dermamnist_256" "organcmnist_256" "organsmnist_256")

for dataset_name in "${datasets[@]}"; do
    dataset_dir="./data/${dataset_name}"
    for seed in "${seeds[@]}"; do
        accelerate launch train_wrn.py \
            --outdir=wrn-runs/${dataset_name}/baseline \
            --train_dir=${dataset_dir}/train \
            --val_dir=${dataset_dir}/val \
            --test_dir=${dataset_dir}/test \
            --batch_size=128 \
            --cond=1 \
            --num_epochs=200 \
            --accum_steps=1 \
            --decay_epochs=60,120,160 \
            --decay_rate=0.2 \
            --warmup_steps=1000 \
            --depth=28 \
            --width_factor=12 \
            --dropout_rate=0.3 \
            --lr=1e-4 \
            --seed=$seed \
            --eval_interval=5 \
            --exp_type=baseline \
            --train_on_latents=1
    done
done