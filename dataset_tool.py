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
import medmnist
from medmnist import INFO, Evaluator

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


def save_dataset(images, labels, dest, dataset_name):
    print(f"Saving dataset to {dest}")
    dest = dest.replace(".zip", "")

    # Clear directory if it exists
    if os.path.exists(dest):
        print(f"Clearing existing directory: {dest}")
        for file in os.listdir(dest):
            file_path = os.path.join(dest, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(dest)

    # Create a dictionary for labels
    label_dict = {}

    # Verify lengths match
    assert len(images) == len(labels), f"Number of images ({len(images)}) doesn't match number of labels ({len(labels)})"

    # Save images
    for idx, (img, label) in tqdm(enumerate(zip(images, labels)), total=len(images)):
        idx_str = f"{idx:08d}"
        archive_fname = f"img{idx_str}.png"
        file_path = os.path.join(dest, archive_fname)

        if img is not None:
            channels = img.shape[0]
            img = PIL.Image.fromarray(img, {1: "L", 3: "RGB"}[channels])
            img.save(file_path, format="png", optimize=False)

            # Store label in dictionary format
            label_dict[archive_fname] = int(label)

    print(f"Processed {len(label_dict)} images and labels")

    # Verify all images have labels
    image_files = [f for f in os.listdir(dest) if f.endswith(".png")]
    assert len(image_files) == len(label_dict), f"Number of saved images ({len(image_files)}) doesn't match number of labels ({len(label_dict)})"

    # Save metadata
    metadata = {"labels": label_dict}
    with open(os.path.join(dest, "dataset.json"), "w") as f:
        json.dump(metadata, f)


def process_medmnist(dataset_name: str, resolution: int, output_dir: str):
    # Get dataset info and class
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])

    # Define transform
    transform = transforms.Compose([transforms.ToTensor()])

    # Load datasets
    train_dataset = DataClass(split="train", transform=transform, download=True, size=resolution, mmap_mode="r")
    val_dataset = DataClass(split="val", transform=transform, download=True, size=resolution, mmap_mode="r")
    test_dataset = DataClass(split="test", transform=transform, download=True, size=resolution, mmap_mode="r")

    # Function to transform images for saving
    def transform_image(img):
        # Convert from tensor to numpy array
        # Assuming img is [C,H,W] tensor in range [0,1]
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return img_np

    # Process and save each split
    for split, dataset in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
        # Extract images and labels
        images = []
        labels = []
        for img, label in dataset:
            images.append(img)
            labels.append(label.item())  # Convert tensor to integer

        # Create output directory for this split
        split_dir = os.path.join(output_dir, dataset_name, split)
        os.makedirs(os.path.dirname(split_dir), exist_ok=True)

        # Save the dataset
        save_dataset(images=images, labels=labels, dest=split_dir, transform_image=transform_image, dataset_name=dataset_name)


@click.command()
@click.option("--dataset", help="Dataset to download and process", required=True)
@click.option("--dest", help="Output directory or archive name", metavar="PATH", type=str, required=True)
@click.option("--transform", help="Input crop/resize mode", metavar="MODE", type=click.Choice(["center-crop", "center-crop-wide"]))
@click.option("--resolution", help="Image size", type=int)
@click.option("--val_ratio", help="Ratio of validation set (0.0-1.0). If 0 or None, no validation set is created", type=float, default=0.2)
def main(dataset: str, dest: str, transform: Optional[str], resolution: Optional[int], val_ratio: Optional[float]):

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
    else:
        SUPPORTED_MEDMNIST = list(INFO.keys())
        assert dataset in SUPPORTED_MEDMNIST, f"Unsupported dataset: {dataset}"
        process_medmnist(dataset, resolution, dest)
        return

    # Ensure val_ratio is within bounds
    val_ratio = min(max(val_ratio, 0.0), 1.0)
    train_size = int((1 - val_ratio) * len(trainset))

    # Create indices for train/val split
    indices = torch.randperm(len(trainset))
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]

    # Get actual data using indices
    train_images = np.array([np.array(trainset[i][0]) for i in train_indices])
    train_labels = np.array([trainset[i][1] for i in train_indices])
    valid_images = np.array([np.array(trainset[i][0]) for i in valid_indices])
    valid_labels = np.array([trainset[i][1] for i in valid_indices])

    # Save train dataset
    save_dataset(train_images, train_labels, os.path.join(dest, "train"), "train")

    # Save validation dataset if it exists
    if val_ratio > 0:
        save_dataset(valid_images, valid_labels, os.path.join(dest, "valid"), "valid")

    # Convert test data to numpy arrays
    test_images = np.array([np.array(img) for img, _ in testset])
    test_labels = np.array([label for _, label in testset])

    # Save test dataset
    save_dataset(test_images, test_labels, os.path.join(dest, "test"), "test")


if __name__ == "__main__":
    main()
