import csv
import os

import torch
from diffusers import AutoencoderKL
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import dnnlib
from training.ece import ECELoss
from training.patch import get_patches
from training.utils import Meter


class Tester:
    """Tester for Wide Residual Networks."""

    def __init__(self, outdir: str, test_dataset_kwargs, network_kwargs, train_on_latents=False):
        """
        Args:
            outdir (str): Output directory.
            test_dataset_kwargs (dict): Test dataset options.
            network_kwargs (dict): Model options.
            train_on_latents (bool, optional, defaults to `False`): Whether to train on latents.
        """
        self.outdir = outdir
        self.test_dataset_kwargs = test_dataset_kwargs
        self.batch_size = 256
        self.network_kwargs = network_kwargs

        self.ece = ECELoss(n_bins=10)

        self.train_on_latents = train_on_latents
        self.img_vae = None
        self.latent_scale_factor = 0.18215

        self._init_tester()

    def _init_tester(self):
        """Initialize the Tester"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prepare_dataloaders()
        self._build_network()

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

    def _set_requires_grad(self, model: torch.nn.Module, requires_grad: bool):
        """Set requires_grad for all parameters in the model.

        Args:
            model (torch.nn.Module): The model to set `requires_grad` for.
            requires_grad (bool): Whether to set `requires_grad` to true or false.
        """
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _build_network(self):
        """Setup network"""
        print("Constructing network...")
        self.net = dnnlib.util.construct_class_by_name(**self.network_kwargs).to(self.device)
        self.ema = EMA(self.net, beta=0.9999, power=3 / 4).to(self.device)

        if self.train_on_latents:
            self.img_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device)
            self.img_vae.eval()
            self._set_requires_grad(self.img_vae, False)

    @torch.no_grad()
    def _encode_latents(self, images: torch.Tensor):
        """Encode the given images to compressed latent space.

        Args:
            images (`torch.Tensor`): The images to encode.

        Returns:
            The encoded latents.
        """

        self.channel_expand = torch.nn.Conv2d(1, 3, kernel_size=1).to(self.device)

        if images.shape[1] == 1:
            images = self.channel_expand(images)

        images = self.img_vae.encode(images)["latent_dist"].sample()
        latents = self.latent_scale_factor * images

        return latents

    def test(self, ckpt_list: list[str]):
        """Test the model given a list of checkpoints."""

        if os.path.exists(os.path.join(self.outdir, "test_metrics.csv")):
            os.remove(os.path.join(self.outdir, "test_metrics.csv"))

        for ckpt in ckpt_list:
            self._load_checkpoint(ckpt)
            metrics = self.evaluate(self.net, self.test_dataloader)
            ckpt_name = os.path.basename(ckpt).split(".")[0]
            self._save_metrics(ckpt_name, metrics)

    def _save_metrics(self, ckpt_name: str, metrics: dict):
        """Save metrics to a CSV file.

        Args:
            ckpt_name (str): Checkpoint name.
            metrics (dict): Metrics to save.
        """
        csv_path = os.path.join(self.outdir, "test_metrics.csv")
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["checkpoint"] + list(metrics.keys()))
            writer.writerow([ckpt_name] + list(metrics.values()))

    def _load_checkpoint(self, ckpt: str):
        """Load checkpoint."""
        data = torch.load(ckpt, weights_only=True)
        self.ema.load_state_dict(data["ema"])
        self.ema.ema_model.eval()


class WRNTester(Tester):
    """Tester for Wide Residual Networks."""

    def __init__(self, outdir: str, test_dataset_kwargs, network_kwargs, train_on_latents=False):
        super().__init__(outdir, test_dataset_kwargs, network_kwargs, train_on_latents)

    @torch.no_grad()
    def evaluate(self, net: torch.nn.Module, dataloader: DataLoader):
        """Evaluate the model on the test dataset.

        Args:
            net (torch.nn.Module): Model to test.
            dataloader (DataLoader): DataLoader for the test dataset.

        Returns:
            dict: Test metrics.
        """
        loss_meter = Meter()
        acc_meter = Meter()
        ece_meter = Meter()

        net.eval()

        with tqdm(total=len(dataloader), desc="Testing", dynamic_ncols=True) as pbar:
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                pbar.update(1)

                labels = labels.argmax(dim=1)

                if self.train_on_latents:
                    images = self._encode_latents(images)

                logits = net(images)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                acc = (logits.argmax(dim=1) == labels).float().mean()

                loss_meter.update(loss.mean(), images.size(0))
                acc_meter.update(acc, images.size(0))
                ece_meter.update(self.ece(logits, labels))

            metrics = {
                "test_cls_loss": loss_meter.compute().item(),
                "test_cls_acc": acc_meter.compute().item(),
                "test_cls_ece": ece_meter.compute().item(),
            }
            pbar.set_postfix(metrics)
            pbar.close()

        return metrics


class EGCTester(Tester):
    """Tester for EGC."""

    def __init__(self, outdir: str, test_dataset_kwargs, network_kwargs, train_on_latents=False):
        super().__init__(outdir, test_dataset_kwargs, network_kwargs, train_on_latents)

    def _build_network(self):
        """Setup network"""
        print("Constructing network...")
        self.net = dnnlib.util.construct_class_by_name(**self.network_kwargs).to(self.device)
        self.ema = EMA(self.net, beta=0.9999, power=3 / 4).to(self.device)

        if self.train_on_latents:
            self.img_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device)
            self.img_vae.eval()
            self._set_requires_grad(self.img_vae, False)
            self.img_resolution, self.img_channels = self.img_resolution // 8, 4

    @torch.no_grad()
    def evaluate(self, net: torch.nn.Module, dataloader: DataLoader):
        """Evaluate the model on the test dataset.

        Args:
            net (torch.nn.Module): Model to test.
            dataloader (DataLoader): DataLoader for the test dataset.

        Returns:
            dict: Test metrics.
        """
        loss_meter = Meter()
        acc_meter = Meter()
        ece_meter = Meter()

        net.eval()

        with tqdm(total=len(dataloader), desc="Testing", dynamic_ncols=True) as pbar:
            for images, labels in dataloader:
                pbar.update(1)
                images, labels = images.to(self.device), labels.to(self.device)

                if self.train_on_latents:
                    images = self._encode_latents(images)

                images, labels = get_patches(images, self.img_resolution), labels.argmax(dim=1)
                clean_timesteps = torch.zeros(images.size(0), dtype=torch.long, device=self.device)

                logits = self.ema(images, clean_timesteps, cls_mode=True)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                acc = (logits.argmax(dim=1) == labels).float().mean()

                loss_meter.update(loss.mean(), images.size(0))
                acc_meter.update(acc, images.size(0))
                ece_meter.update(self.ece(logits, labels))

            metrics = {
                "test_cls_loss": loss_meter.compute().item(),
                "test_cls_acc": acc_meter.compute().item(),
                "test_cls_ece": ece_meter.compute().item(),
            }
            pbar.set_postfix(metrics)
            pbar.close()

        return metrics
