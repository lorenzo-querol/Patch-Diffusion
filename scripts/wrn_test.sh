#!/bin/bash

python test_wrn.py \
    --outdir=wrn-runs/00001-run \
    --test_dir=./data/bloodmnist_28/test \
    --batch_size=128 \
    --cond=1 \
    --seed=1

