## Patch Diffusion: Faster and More Data-Efficient Training of Diffusion Models<br><sub>Official PyTorch implementation</sub>

![Teaser image](./docs/patch_diffusion_illustration.png)

**Patch Diffusion: Faster and More Data-Efficient Training of Diffusion Models**<br>
Zhendong Wang, Yifan Jiang, Huangjie Zheng, Peihao Wang, Pengcheng He, Zhangyang Wang, Weizhu Chen, Mingyuan Zhou
<br>https://arxiv.org/abs/2304.12526 <br>

Abstract: *Diffusion models are powerful, but they require a lot of time and data to train. We propose **Patch Diffusion**, a generic patch-wise training framework, to significantly reduce the training time costs while improving data efficiency, which thus helps democratize diffusion model training to broader users. At the core of our innovations is a new conditional score function at the patch level, where the patch location in the original image is included as additional coordinate channels, while the patch size is randomized and diversified throughout training to encode the cross-region dependency at multiple scales. Sampling with our method is as easy as in the original diffusion model. Through Patch Diffusion, we could achieve 2x faster training, while maintaining comparable or better generation quality. Patch Diffusion meanwhile improves the performance of diffusion models trained on relatively small datasets, e.g., as few as 5,000 images to train from scratch. We achieve state-of-the-art FID scores 1.77 on CelebA-64x64 and 1.93 on AFHQv2-Wild-64x64. We share our code and pre-trained models here.*


## Requirements

* We build our Patch Diffusion upon the [EDM](https://github.com/NVlabs/edm) code base and the python environment set up is similar.
* Python libraries: See [environment.yml](./environment.yml) for exact library dependencies. You can use the following commands with Miniconda3 to create and activate your Python environment:
  - `conda env create -f environment.yml -n edm`
  - `conda activate edm`
* Docker users:
  - Ensure you have correctly installed the [NVIDIA container runtime](https://docs.docker.com/config/containers/resource_constraints/#gpu).
  - Use the [provided Dockerfile](./Dockerfile) to build an image with the required library dependencies.

## Getting started

### Preparing datasets

Download the dataset that you want to try on, such as [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [FFHQ](https://github.com/NVlabs/ffhq-dataset), [LSUN](https://github.com/fyu/lsun), [ImageNet](https://image-net.org/index.php).
Resize the image to the desired resolution, e.g., 64x64 here, and compute the fid-reference file for the preparation of fid computation, as follows:
```.bash
python dataset_tool.py --source=downloads/{dataset_folder} \
    --dest=datasets/{data_name}.zip --resolution=64x64 --transform=center-crop
python fid.py ref --data=datasets/{data_name}.zip --dest=fid-refs/{data_name}-64x64.npz

For CIFAR10 dataset, you can use the following commands. Make sure you are in the repository root directory.

wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

python dataset_tool.py --source=../cifar-10-python.tar.gz --dest ../data/cifar10/train --transform center-crop --resolution 32x32

python dataset_tool.py --source=../cifar-10-python.tar.gz --dest ../data/cifar10/test --transform center-crop --resolution 32x32 --is_test

python dataset_tool.py --dataset cifar10 --dest data/cifar100 --transform center-crop --resolution 32x32 --val_ratio 0.2

```

### Train Patch Diffusion

You can train new models using `train.py`. For example:

```.bash
torchrun --standalone --nproc_per_node=2 train.py \
    --outdir=training-runs \
    --train_dir=data/cifar100/train \
    --val_dir=data/cifar100/valid \
    --cond=1 \
    --arch=ebm \
    --cres=1,2,2 \
    --attn_resolutions=16,8 \
    --batch=128 \
    --lr=2e-4 \
    --dropout=0.0 \
    --augment=0.0 \
    --real_p=0.5 \
    --seed=1
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

You can generate images using `generate.py`. For example:
```.bash
# For DDPM++ Architecture we use
torchrun --standalone --nproc_per_node=8 generate.py --steps=50 --resolution 64 --batch 64 --outdir=fid-tmp --seeds=0-49999 --subdirs --network=/path-to-the-pkl/

# For ADM Architecture we use
torchrun --standalone --nproc_per_node=8 generate.py --steps=256 --S_churn=40 --S_min=0.05 --S_max=50 --S_noise=1.003 --resolution 32 --on_latents=1 --batch 64 --outdir=fid-tmp --seeds=0-49999 --subdirs --network=/path-to-the-pkl/

# CUSTOM For ADM Architecture we use
torchrun --standalone --nproc_per_node=2 generate.py --steps=256 --S_churn=40 --S_min=0.05 --S_max=50 --S_noise=1.003 --resolution 32 --batch 64 --outdir=fid-tmp --seeds=0-49999 --subdirs --network=training-runs/00000-train-cond-ebm-pedm-gpus2-batch128-fp32/network-snapshot-007536.pkl --cfg=1.3
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


## Citation

```
@article{wang2023patch,
  title={Patch Diffusion: Faster and More Data-Efficient Training of Diffusion Models},
  author={Wang, Zhendong and Jiang, Yifan and Zheng, Huangjie and Wang, Peihao and He, Pengcheng and Wang, Zhangyang and Chen, Weizhu and Zhou, Mingyuan},
  journal={arXiv preprint arXiv:2304.12526},
  year={2023}
}
```

## Acknowledgments

We thank [EDM](https://github.com/NVlabs/edm) authors for providing the great code base and [HuggingFace](https://huggingface.co/) for providing the easy access of Stable Diffusion Auto-Encoders.
