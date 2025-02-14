#!/bin/bash

seeds=(0)
model_type=(wrn)
strategies=(baseline)
ckpt_types=(best final)
dataset="bloodmnist_256"
dataset_dir="./data/${dataset}"

for seed in "${seeds[@]}"; do
    for strategy in "${strategies[@]}"; do
        for ckpt_type in "${ckpt_types[@]}"; do
            python test_wrn.py \
                --model_type=${model_type} \
                --outdir=./${model_type}-runs/${dataset}/${strategy}/0000${seed}-run \
                --test_dir=${dataset_dir}/test \
                --ckpt_type=${ckpt_type} \
                --train_on_latents=1
        done
    done
done