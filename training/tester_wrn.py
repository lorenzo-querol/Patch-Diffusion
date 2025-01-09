import csv
import os

import torch
from accelerate.utils import set_seed
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import dnnlib
from training.ece import ECELoss


class Meter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, count=1):
        self.total += value * count
        self.count += count

    def compute(self):
        return self.total / self.count


class Tester:
    def __init__(
        self,
        outdir="./test_results",  # Output directory
        test_dataset_kwargs={},  # Training dataset options
        network_kwargs={},  # Model options
        batch_size=128,  # Batch size
        seed=1,  # Seed for reproducibility
    ):
        self.outdir = outdir
        self.test_dataset_kwargs = test_dataset_kwargs
        self.network_kwargs = network_kwargs
        self.seed = seed
        self.batch_size = batch_size

        self.ece = ECELoss(n_bins=10)
        self._init_tester()

    def _init_tester(self):
        """Initialize the Trainer: seeds, datasets, and network."""
        self._init_env()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prepare_dataloaders()
        self._build_network()

    def _init_env(self):
        set_seed(self.seed)
        torch.backends.cudnn.benchmark = True

    def _prepare_dataloaders(self):
        """Prepare datasets and dataloaders."""
        print("Loading test dataset...")

        dataset_obj = dnnlib.util.construct_class_by_name(**self.test_dataset_kwargs)
        self.img_resolution, self.img_channels, self.label_dim = (
            dataset_obj.resolution,
            dataset_obj.num_channels,
            dataset_obj.label_dim,
        )
        del dataset_obj

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5] * self.img_channels, std=[0.5] * self.img_channels)])
        test_dataset = dnnlib.util.construct_class_by_name(**self.test_dataset_kwargs, transform=transform)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=4)

    def _build_network(self):
        """Setup network"""
        print("Constructing network...")
        self.net = dnnlib.util.construct_class_by_name(**self.network_kwargs)
        self.ema = EMA(self.net)

        self.net = self.net.to(self.device)
        self.ema = self.ema.to(self.device)

    def test(self, ckpt_list: list):
        """Test the model on the test set."""

        if os.path.exists(os.path.join(self.outdir, "test_metrics.csv")):
            os.remove(os.path.join(self.outdir, "test_metrics.csv"))

        for ckpt in ckpt_list:
            self._load(ckpt)
            metrics = self.test_ckpt(self.net, self.test_dataloader)
            ckpt_name = os.path.basename(ckpt).split(".")[0]
            self._save_metrics(ckpt_name, metrics)

    @torch.no_grad()
    def test_ckpt(self, net: torch.nn.Module, dataloader: DataLoader):
        loss_meter = Meter()
        acc_meter = Meter()
        ece_meter = Meter()

        net.eval()

        with tqdm(total=len(dataloader), desc="Testing", dynamic_ncols=True) as pbar:
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                pbar.update(1)

                y = y.argmax(dim=1)

                logits = net(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                acc = (logits.argmax(dim=1) == y).float().mean()

                loss_meter.update(loss.mean(), x.size(0))
                acc_meter.update(acc, x.size(0))
                ece_meter.update(self.ece(logits, y))

            metrics = {
                "test_cls_loss": loss_meter.compute().item(),
                "test_cls_acc": acc_meter.compute().item(),
                "test_cls_ece": ece_meter.compute().item(),
            }
            pbar.set_postfix(metrics)
            pbar.close()

        return metrics

    def _save_metrics(self, ckpt_name: str, metrics: dict):
        """Save metrics to CSV file."""
        csv_path = os.path.join(self.outdir, "test_metrics.csv")
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["checkpoint"] + list(metrics.keys()))
            writer.writerow([ckpt_name] + list(metrics.values()))

    def _load(self, ckpt: str):
        """Either load the latest checkpoint from the directory or load the checkpoint from the file

        :params resume_from: The directory or the file to resume from. If it is a directory, the latest
        checkpoint will be loaded. Else, the specified file will be loaded.
        """
        data = torch.load(ckpt, weights_only=True)
        self.net.load_state_dict(data["net"])
        # self.ema.load_state_dict(data["ema"])
