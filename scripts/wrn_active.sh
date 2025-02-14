#!/bin/bash

seeds=(1)
datasets=(dermamnist_256)
strategies=(lc)

for dataset_name in "${datasets[@]}"; do
    dataset_dir="./data/${dataset_name}"
    for seed in "${seeds[@]}"; do
        for strategy in "${strategies[@]}"; do
            accelerate launch train_wrn.py \
                --outdir=wrn-runs/${dataset_name}/${strategy} \
                --train_dir=${dataset_dir}/train \
                --val_dir=${dataset_dir}/val \
                --test_dir=${dataset_dir}/test \
                --batch_size=128 \
                --cond=1 \
                --num_epochs=20 \
                --accum_steps=1 \
                --warmup_steps=0 \
                --depth=28 \
                --width_factor=12 \
                --dropout_rate=0.3 \
                --lr=1e-4 \
                --optimizer=adam \
                --use_bn=0 \
                --seed=${seed} \
                --eval_interval=1 \
                --exp_type=active \
                --num_samples=0.1 \
                --strategy=${strategy} \
                --train_on_latents=1
        done
    done
done