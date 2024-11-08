import csv
import json
import os
import random
import PIL
import numpy as np
import pyspng
import torch.backends
import click
from calibration.ece import ECELoss
import dnnlib
import torch
import torch.nn.functional as F
import torchvision as tv
from training.dataset import Dataset


def init_environment(seed):
    if seed is None:
        seed = random.randint(0, 2**31)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True


def process_batch(net, optim, dataloader, device, training=False):
    total_loss = 0
    total_correct = 0
    total_ece = 0
    total_samples = 0

    ece_criterion = ECELoss(n_bins=20)

    net.train(training)

    with torch.enable_grad() if training else torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            logits = net(images)
            loss = F.cross_entropy(logits, labels)
            ece = ece_criterion(logits, labels.argmax(1))

            if training:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(1) == labels.argmax(1)).sum().item()
            total_ece += ece.item() * images.size(0)
            total_samples += images.size(0)

    metrics = {"loss": total_loss / total_samples, "acc": total_correct / total_samples, "ece": total_ece / total_samples}
    return metrics


def get_next_run_dir(base_dir, desc="wrn_run"):
    # If directory doesn't exist, start with run 0
    os.makedirs(base_dir, exist_ok=True)

    # Get all numeric prefixes of existing directories
    existing_runs = [int(d[:5]) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d[:5].isdigit()]

    # Get next run number
    next_run = max(existing_runs, default=-1) + 1
    return os.path.join(base_dir, f"{next_run:05d}-{desc}")


class WRNDataset(Dataset):
    def __init__(
        self,
        path,  # Path to directory or zip.
        resolution=None,  # Ensure specific resolution, None = highest available.
        use_pyspng=True,  # Use pyspng if available?
        transforms=None,  # Optional image transform.
        **super_kwargs,  # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None
        self.transforms = transforms

        if os.path.isdir(self._path):
            self._type = "dir"
            self._all_fnames = {
                os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files
            }
        else:
            raise IOError("Path must point to a directory or zip")

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError("No image files found in the specified path")

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError("Image files do not match the specified resolution")
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _open_file(self, fname):
        return open(os.path.join(self._path, fname), "rb")

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if self._use_pyspng and pyspng is not None and self._file_ext(fname) == ".png":
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = "dataset.json"
        if fname not in self._all_fnames:
            return None

        with self._open_file(fname) as f:
            labels = json.load(f)["labels"]

        if labels is None:
            return None

        labels = dict(labels)
        labels = [labels[fname.replace("\\", "/")] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

        return labels

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)

        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image

        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8

        if self.transforms is not None:
            image = PIL.Image.fromarray(image.transpose(1, 2, 0))  # CHW -> HWC
            image = self.transforms(image)

        return image, self.get_label(idx)


mean = {
    "cifar10": (0.4914, 0.4822, 0.4465),
    "cifar100": (0.5071, 0.4867, 0.4408),
}

std = {
    "cifar10": (0.2023, 0.1994, 0.2010),
    "cifar100": (0.2675, 0.2565, 0.2761),
}


@click.command()
@click.option("--dataset", help="Dataset to use", metavar="STR", required=True)
@click.option("--outdir", help="Where to save the results", metavar="DIR", type=str, required=True)
@click.option("--train_dir", help="Path to the train dataset", metavar="ZIP|DIR", type=str, required=True)
@click.option("--val_dir", help="Path to the valid dataset", metavar="ZIP|DIR", type=str, required=True)
@click.option("--test", help="Test the model", is_flag=True)
@click.option("--test_dir", help="Path to the test dataset", metavar="ZIP|DIR", type=str)
@click.option("--network_path", help="Path to the network", metavar="STR", type=str)

# WRN-related.
@click.option("--num_epochs", help="Number of epochs", metavar="INT", type=int, required=True, default=200)
@click.option("--batch", help="Total batch size", metavar="INT", type=click.IntRange(min=1), required=True, default=128)
@click.option("--lr", help="Learning rate", metavar="FLOAT", type=float, required=True, default=0.1)
@click.option("--depth", help="Depth of the network", metavar="INT", type=int, default=28)
@click.option("--width", help="Width of the network", metavar="INT", type=int, default=10)
@click.option("--norm", help="Normalization layer", metavar="STR", type=str, default="batch")
@click.option("--dropout", help="Dropout rate", metavar="FLOAT", type=float, default=0.3)

# Evaluation-related.
@click.option("--eval_every", help="How often to evaluate the model", metavar="INT", type=click.IntRange(min=1), default=5, show_default=True)

# I/O-related.
@click.option("--seed", help="Random seed  [default: random]", metavar="INT", type=int)
@click.option("--snap", help="How often to save snapshots", metavar="TICKS", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--workers", help="Number of data loading workers", metavar="INT", type=click.IntRange(min=1), default=1, show_default=True)
def main(**kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opts = dnnlib.EasyDict(kwargs)

    init_environment(opts.seed)

    # Create numbered run directory
    run_dir = get_next_run_dir(opts.outdir)
    os.makedirs(run_dir, exist_ok=False)  # Will raise error if dir exists

    # Initialize config dict.
    c = dnnlib.EasyDict()

    print(f"Starting run in {run_dir}")
    print("Loading dataset...")

    dataset_kwargs = dnnlib.EasyDict(class_name="train_wrn.WRNDataset", path=opts.train_dir, use_labels=True)
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    resolution = dataset_obj.resolution
    del dataset_obj, dataset_kwargs

    transforms = {
        "train": tv.transforms.Compose(
            [
                tv.transforms.RandomCrop(resolution, padding=4),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean[opts.dataset], std[opts.dataset]),
            ]
        ),
        "test": tv.transforms.Compose(
            [
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean[opts.dataset], std[opts.dataset]),
            ]
        ),
    }

    c.dataset_kwargs = dnnlib.EasyDict(class_name="train_wrn.WRNDataset", path=opts.train_dir, use_labels=True, transforms=transforms["train"])
    c.val_dataset_kwargs = dnnlib.EasyDict(class_name="train_wrn.WRNDataset", path=opts.val_dir, use_labels=True, transforms=transforms["test"])
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)

    # Create dataloaders.
    dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
    val_dataset_obj = dnnlib.util.construct_class_by_name(**c.val_dataset_kwargs)
    train_dataloader = torch.utils.data.DataLoader(dataset=dataset_obj, batch_size=opts.batch, **c.data_loader_kwargs)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset_obj, batch_size=opts.batch, **c.data_loader_kwargs)

    img_channels, num_classes = dataset_obj.num_channels, dataset_obj.label_dim

    # Initialize network and optimizer.
    print("Loading network...")
    c.network_kwargs = dnnlib.EasyDict(
        class_name="baseline_models.wide_resnet.Wide_ResNet",
        depth=opts.depth,
        widen_factor=opts.width,
        in_channels=img_channels,
        num_classes=num_classes,
        dropout_rate=opts.dropout,
        norm=opts.norm,
    )
    c.optimizer_kwargs = dnnlib.EasyDict(class_name="torch.optim.SGD", lr=opts.lr, momentum=0.9, weight_decay=5e-4)
    net = dnnlib.util.construct_class_by_name(**c.network_kwargs).to(device)
    optim = dnnlib.util.construct_class_by_name(**c.optimizer_kwargs, params=net.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[60, 120, 160], gamma=0.2)

    # Create output directory.
    print("Creating output directory...")

    # remove transforms from c dataset kwargs
    c.dataset_kwargs.pop("transforms")
    c.val_dataset_kwargs.pop("transforms")

    logger = dnnlib.util.Logger(file_name=os.path.join(run_dir, "log.txt"), file_mode="a", should_flush=True)
    with open(os.path.join(run_dir, "training_options.json"), "wt") as f:
        json.dump(c, f, indent=2)

    metrics_file = os.path.join(run_dir, "metrics.csv")
    with open(metrics_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["epoch", "train_loss", "train_acc", "train_ece", "val_loss", "val_acc", "val_ece"])
        writer.writeheader()

    if opts.test:
        assert opts.test_dir is not None, "Must provide a test directory for testing"
        assert opts.network_path is not None, "Must provide a network path for testing"

        c.test_dataset_kwargs = dnnlib.EasyDict(class_name="train_wrn.WRNDataset", path=opts.test_dir, use_labels=True, transforms=transforms["test"])
        test_dataset_obj = dnnlib.util.construct_class_by_name(**c.test_dataset_kwargs)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset_obj, batch_size=opts.batch, **c.data_loader_kwargs)

        print("Testing...")
        net.load_state_dict(torch.load(opts.network_path)["model_state_dict"])
        test_metrics = process_batch(net, optim, test_dataloader, device)
        test_str = f"Test Loss: {test_metrics['loss']:.4f} | Test Acc: {test_metrics['acc']:.4f} | Test ECE: {test_metrics['ece']:.4f}\n"
        print(test_str)
        return

    print("Training...")
    for epoch in range(opts.num_epochs):
        train_metrics = process_batch(net, optim, train_dataloader, device, training=True)
        train_str = f"Epoch {epoch+1} | Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['acc']:.4f} | ECE: {train_metrics['ece']:.4f}\n"
        logger.write(train_str)

        if epoch % opts.eval_every == 0:
            val_metrics = process_batch(net, optim, val_dataloader, device)
            val_str = f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['acc']:.4f} | Val ECE: {val_metrics['ece']:.4f}\n"
            logger.write(val_str)

        if epoch % opts.snap == 0:
            dict_values = {"model_state_dict": net.state_dict(), "optim_state_dict": optim.state_dict()}
            torch.save(dict_values, os.path.join(run_dir, f"network-snapshot-epoch-{epoch}.pt"))

        with open(metrics_file, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["epoch", "train_loss", "train_acc", "train_ece", "val_loss", "val_acc", "val_ece"])
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "train_ece": train_metrics["ece"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_ece": val_metrics["ece"],
            }
            writer.writerow(epoch_metrics)

        scheduler.step()

    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optim_state_dict": optim.state_dict(),
        },
        os.path.join(run_dir, "network-final.pt"),
    )


if __name__ == "__main__":
    main()
