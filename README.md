## Getting started

### Preparing datasets

For CIFAR10/100 dataset, you can use the following commands. 

```bash
# Example: 20% of the training data will be used as validation set if val_ratio is set to 0.2
python dataset_tool.py --dataset cifar10 \
                       --dest ./data \
                       --val_ratio 0.0
```

For MedMNIST datasets

```bash
python dataset_tool.py --dataset bloodmnist \
                       --dest ./data \
                       --resolution 64
```

### Train Patch EGC

You can train new models using `train.py`. For example with CIFAR10:

```shell
accelerate launch train.py \
        --outdir=training-runs \
        --train_dir=./data/cifar10/train \
        --val_dir=./data/cifar10/test \
        --batch_size=128 \
        --cond=1 \
        --model_channels=128 \
        --num_res_blocks=2 \
        --channel_mult=1,2,2,2 \
        --attn_resolutions=16,8 \
        --dropout_rate=0.1 \
        --schedule_name=cosine \
        --timesteps=1000 \
        --lr=2e-4 \
        --ce_weight=0.0 \
        --label_smooth=0.2 \
        --eval_interval=10 \
        --resume_from=training-runs/00004-run
```

You can train new baseline models using `train_wrn.py`. For example:

```.bash
CUDA_VISIBLE_DEVICES=1 python train_wrn.py \
    --dataset=cifar10 \
    --outdir=wrn-runs \
    --train_dir=../data/cifar10/train \
    --val_dir=../data/cifar10/test \
    --batch=128 \
    --norm=batch \
    --eval_every=10 \
    --seed=1
```
To test the performance of the WRN model, you can use the following command:

```.bash
CUDA_VISIBLE_DEVICES=1 python train_wrn.py \
    --dataset=cifar10 \
    --outdir=wrn-test-runs \
    --train_dir=../data/cifar10/train \
    --val_dir=../data/cifar10/valid \
    --batch=128 \
    --norm=batch \
    --seed=1 \
    --test \
    --test_dir=../data/cifar10/test \
    --network_path=wrn-runs/00027-wrn_run/network-final.pt
```

We follow the hyperparameter settings of EDM, and introduce two new parameters here:

- `--real_p`: the ratio of full size image used in the training.
- `--train_on_latents`: where to train on the Latent Diffusion latent space, instead of the pixel space. Note we trained our models on the latent space for 256x256 images. 

### Inference Patch Diffusion
USE THIS!

```shell
python sample.py \
        --out_dir=samples \
        --train_dir=./data/cifar10/train \
        --model_dir=training-runs/00022-run \
        --batch_size=128 \
        --cond=1 \
        --num_samples=100 \
        --classifier_scale=6.0 \
        --model_channels=192 \
        --num_res_blocks=3 \
        --channel_mult=1,2,2 \
        --attn_resolutions=16,8 \
        --dropout_rate=0.1 \
        --schedule_name=cosine \
        --timesteps=1000
```

You can generate images using `generate.py`. For example:
```.bash
# For DDPM++ Architecture we use
torchrun --standalone --nproc_per_node=8 generate.py --steps=50 --resolution 64 --batch 64 --outdir=fid-tmp --seeds=0-49999 --subdirs --network=/path-to-the-pkl/

# For ADM Architecture we use
torchrun --standalone --nproc_per_node=8 generate.py --steps=256 --S_churn=40 --S_min=0.05 --S_max=50 --S_noise=1.003 --resolution 32 --on_latents=1 --batch 64 --outdir=fid-tmp --seeds=0-49999 --subdirs --network=/path-to-the-pkl/

# CUSTOM For ADM Architecture we use
torchrun --standalone --nproc_per_node=2 generate.py --steps=256 --S_churn=40 --S_min=0.05 --S_max=50 --S_noise=1.003 --resolution 32 --batch 64 --outdir=fid-tmp --seeds=0-49999 --subdirs --network=training-runs/00042-train-cond-ebm-pedm-gpus2-batch256-fp32/network-snapshot-003072.pkl --cfg=1.3
```

We share our pretrained model checkpoints at [Huggingface Page](https://huggingface.co/zhendongw/patch-diffusion/tree/main). For ImageNet dataset, to generate with classifier-free-guidance, please add `--cfg=1.3` to the command. 

### Calculating FID

To compute Fr&eacute;chet inception distance (FID) for a given model and sampler, first generate 50,000 random images and then compare them against the dataset reference statistics using `fid.py`:

```.bash
# Generate 50000 images and save them as fid-tmp/*/*.png
<!-- torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl -->

torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --network=training-runs/00000-train-cond-ebm-pedm-gpus2-batch128-fp32/network-snapshot-007536.pkl --cfg=1.0


# Calculate FID
torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
```

Both of the above commands can be parallelized across multiple GPUs by adjusting `--nproc_per_node`. The second command typically takes 1-3 minutes in practice, but the first one can sometimes take several hours, depending on the configuration. See [`python fid.py --help`](./docs/fid-help.txt) for the full list of options.


