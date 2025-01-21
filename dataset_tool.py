import json
import os
from typing import Callable, Optional, Tuple, Union

import click
import medmnist
import numpy as np
import PIL.Image
import torch
import torchvision
import torchvision.transforms as transforms
from medmnist import INFO
from tqdm import tqdm

# ----------------------------------------------------------------------------


def save_dataset(images, labels, dest):
    print(f"Saving dataset to {dest}...")

    # Create the output directory
    if os.path.exists(dest):
        for file in os.listdir(dest):
            os.remove(os.path.join(dest, file))
    else:
        os.makedirs(dest)

    label_dict = {}

    for idx, (img, label) in tqdm(enumerate(zip(images, labels)), total=len(images)):
        idx_str = f"{idx:08d}"
        archive_fname = f"img{idx_str}.png"
        file_path = os.path.join(dest, archive_fname)

        # Convert from CHW to HWC format and scale to 0-255 range
        img_np = (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)

        # Save as grayscale or RGB image
        is_grayscale = img_np.shape[-1] == 1

        if is_grayscale:
            img_pil = PIL.Image.fromarray(img_np.squeeze(), "L")
        else:
            img_pil = PIL.Image.fromarray(img_np, "RGB")

        img_pil.save(file_path, format="png")

        label_dict[archive_fname] = int(label.item())

    # Save metadata
    metadata = {"labels": label_dict}
    with open(os.path.join(dest, "dataset.json"), "w") as f:
        json.dump(metadata, f)


@click.command()
@click.option("--dataset", help="Dataset to download and process", required=True)
@click.option("--dest", help="Output directory or archive name", metavar="PATH", type=str, required=True)
@click.option("--val_ratio", help="Ratio of validation set (0.0-1.0). If 0 or None, no validation set is created", type=float, default=0.0)
@click.option("--resolution", help="Resolution of the images", type=int, default=None)
def main(dataset: str, dest: str, val_ratio: Optional[float], resolution: int):

    validset = None
    if dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root="../tmp", train=True, download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR10(root="../tmp", train=False, download=True, transform=transforms.ToTensor())
    elif dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root="../tmp", train=True, download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR100(root="../tmp", train=False, download=True, transform=transforms.ToTensor())
    elif dataset == "mnist":
        trainset = torchvision.datasets.MNIST(root="../tmp", train=True, download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root="../tmp", train=False, download=True, transform=transforms.ToTensor())
    elif dataset in ["bloodmnist", "dermamnist", "organcmnist", "organsmnist"]:
        assert resolution in [28, 64, 128, 224], f"Unsupported resolution: {resolution} for MedMNIST datasets"
        info = INFO[dataset]
        DataClass = getattr(medmnist, info["python_class"])
        resize = 256 if resolution == 224 else resolution
        resize = 32 if resolution == 28 else resolution

        kwargs = {"root": "../tmp", "download": True, "transform": transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])}
        trainset = DataClass(split="train", size=resolution, mmap_mode="r", **kwargs)
        validset = DataClass(split="val", size=resolution, mmap_mode="r", **kwargs)
        testset = DataClass(split="test", size=resolution, mmap_mode="r", **kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    val_ratio = min(max(val_ratio, 0.0), 1.0)
    train_size = int((1 - val_ratio) * len(trainset))

    indices = torch.randperm(len(trainset))
    train_indices = indices[:train_size]

    train_images = np.array([trainset[i][0].numpy() for i in train_indices])
    train_labels = np.array([trainset[i][1] for i in train_indices])

    if val_ratio > 0:
        valid_indices = indices[train_size:]
        valid_images = np.array([trainset[i][0].numpy() for i in valid_indices])
        valid_labels = np.array([trainset[i][1] for i in valid_indices])

    if validset is not None:
        valid_images = np.array([img.numpy() for img, _ in validset])
        valid_labels = np.array([label for _, label in validset])

    test_images = np.array([img.numpy() for img, _ in testset])
    test_labels = np.array([label for _, label in testset])

    if resolution is not None:
        resized_resolutions = {
            28: 32,
            224: 256,
        }

        if resolution in resized_resolutions:
            resolution = resized_resolutions[resolution]

        train_name = f"{dataset}_{resolution}"
        dataset = train_name

    save_dataset(train_images, train_labels, os.path.join(dest, dataset, "train"))

    if val_ratio > 0:
        save_dataset(valid_images, valid_labels, os.path.join(dest, dataset, "val"))

    if validset is not None:
        save_dataset(valid_images, valid_labels, os.path.join(dest, dataset, "val"))

    save_dataset(test_images, test_labels, os.path.join(dest, dataset, "test"))


if __name__ == "__main__":
    main()
