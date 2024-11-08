import os
import json
import io
import zipfile
import numpy as np
import PIL.Image
from typing import Callable, Optional, Tuple, Union
from tqdm import tqdm
import click
import torch
import torchvision
import torchvision.transforms as transforms

# ----------------------------------------------------------------------------


def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a


# ----------------------------------------------------------------------------


def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = dest.split(".")[-1]

    if dest_ext == "zip":
        if os.path.dirname(dest) != "":
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        # Add compression and allowZip64 parameters
        zf = zipfile.ZipFile(
            file=dest,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,  # Use compression
            allowZip64=True,  # Enable support for large files
        )

        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            # Add duplicate checking
            if fname in zf.namelist():
                raise ValueError(f"Duplicate file entry: {fname}")
            zf.writestr(fname, data)

        return "", zip_write_bytes, zf.close


# ----------------------------------------------------------------------------


def save_dataset(images, labels, dest, transform_image, dataset_name):
    print(f"Saving dataset to {dest}")
    # Remove .zip extension if present
    dest = dest.replace(".zip", "")
    os.makedirs(dest, exist_ok=True)

    label_list = []

    # Save images
    for idx, (img, label) in tqdm(enumerate(zip(images, labels)), total=len(images)):
        idx_str = f"{idx:08d}"
        # Save directly to a file
        archive_fname = f"img{idx_str}.png"
        file_path = os.path.join(dest, archive_fname)

        img = transform_image(img)
        if img is None:
            continue

        channels = img.shape[2] if img.ndim == 3 else 1
        img = PIL.Image.fromarray(img, {1: "L", 3: "RGB"}[channels])
        img.save(file_path, format="png", optimize=False)

        label_list.append([archive_fname, int(label)])

    # Save metadata
    metadata = {"labels": label_list if all(x is not None for x in label_list) else None}
    with open(os.path.join(dest, "dataset.json"), "w") as f:
        json.dump(metadata, f)


@click.command()
@click.option("--dataset", help="Dataset to download and process", type=click.Choice(["cifar10", "cifar100", "mnist"]), required=True)
@click.option("--dest", help="Output directory or archive name", metavar="PATH", type=str, required=True)
@click.option("--transform", help="Input crop/resize mode", metavar="MODE", type=click.Choice(["center-crop", "center-crop-wide"]))
@click.option("--resolution", help="Output resolution (e.g., 512x512)", metavar="WxH", type=str)
@click.option("--val_ratio", help="Ratio of validation set (0.0-1.0). If 0 or None, no validation set is created", type=float, default=0.2)
def main(dataset: str, dest: str, transform: Optional[str], resolution: Optional[str], val_ratio: Optional[float]):
    """Download specified dataset and optionally split into train, validation, and test sets."""

    # Download dataset
    if dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root="../tmp", train=True, download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR10(root="../tmp", train=False, download=True, transform=transforms.ToTensor())
    elif dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root="../tmp", train=True, download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR100(root="../tmp", train=False, download=True, transform=transforms.ToTensor())
    elif dataset == "mnist":
        trainset = torchvision.datasets.MNIST(root="../tmp", train=True, download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root="../tmp", train=False, download=True, transform=transforms.ToTensor())

    # Split datasets based on validation ratio
    if val_ratio is None or val_ratio <= 0:
        # No validation set
        train_dataset = trainset
        valid_dataset = None
    else:
        # Ensure val_ratio is within bounds
        val_ratio = min(max(val_ratio, 0.0), 1.0)
        train_size = int((1 - val_ratio) * len(trainset))
        valid_size = len(trainset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(
            trainset,
            [train_size, valid_size],
            generator=torch.Generator().manual_seed(42),
        )

    # Convert training data to numpy arrays
    train_images = np.array([np.array(img) for img, _ in train_dataset])
    train_labels = np.array([label for _, label in train_dataset])

    # Convert test data to numpy arrays
    test_images = np.array([np.array(img) for img, _ in testset])
    test_labels = np.array([label for _, label in testset])

    # Parse resolution
    if resolution is not None:
        resolution = tuple(map(int, resolution.split("x")))
    else:
        resolution = (None, None)

    # Define transform function
    def transform_image(img):
        img = img.transpose(1, 2, 0)  # Convert from CHW to HWC
        img = (img * 255).astype(np.uint8)  # Convert to uint8
        img = PIL.Image.fromarray(img)
        if transform == "center-crop":
            img = img.crop((0, 0, resolution[0], resolution[1]))
        elif transform == "center-crop-wide":
            img = img.crop((0, 0, resolution[0], resolution[1]))
        img = img.resize(resolution, PIL.Image.Resampling.LANCZOS)
        return np.array(img)

    # Save train dataset
    save_dataset(train_images, train_labels, os.path.join(dest, "train"), transform_image, "train")

    # Save validation dataset if it exists
    if valid_dataset is not None:
        valid_images = np.array([np.array(img) for img, _ in valid_dataset])
        valid_labels = np.array([label for _, label in valid_dataset])
        save_dataset(valid_images, valid_labels, os.path.join(dest, "valid"), transform_image, "valid")

    # Save test dataset
    save_dataset(test_images, test_labels, os.path.join(dest, "test"), transform_image, "test")


if __name__ == "__main__":
    main()
