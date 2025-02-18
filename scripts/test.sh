#!/bin/bash

seeds=(0 1 2 3 4)
model_type=(wrn)
strategies=(baseline)
ckpt_types=(best final)
dataset=(organamnist_256)
dataset_dir="./data/${dataset}"

for seed in "${seeds[@]}"; do
    for strategy in "${strategies[@]}"; do
        for ckpt_type in "${ckpt_types[@]}"; do
            python test_wrn.py \
                --model_type=${model_type} \
                --outdir=./results/${model_type}/${dataset}/${strategy}/0000${seed}-run \
                --data_dir=${dataset_dir}/test \
                --ckpt_dir=./${model_type}-runs/${dataset}/${strategy}/0000${seed}-run \
                --ckpt_type=${ckpt_type} \
                --train_on_latents=1
        done
    done
done