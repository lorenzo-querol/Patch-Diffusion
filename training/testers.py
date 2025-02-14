import csv
import os

import numpy as np
from sklearn.manifold import TSNE
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
    """Generic Tester class."""

    def __init__(self, outdir: str, test_dataset_kwargs, network_kwargs, ckpt_type, train_on_latents=False):
        """
        Args:
            outdir (str): Output directory.
            test_dataset_kwargs (dict): Test dataset options.
            network_kwargs (dict): Model options.
            ckpt_type (str): Checkpoint type.
            train_on_latents (bool, optional, defaults to `False`): Whether to train on latents.
        """
        self.outdir = outdir
        self.test_dataset_kwargs = test_dataset_kwargs
        self.batch_size = 128
        self.network_kwargs = network_kwargs
        self.ckpt_type = ckpt_type
        self.train_on_latents = train_on_latents

        self.ece = ECELoss(n_bins=10)
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

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * self.img_channels, std=[0.5] * self.img_channels),
            ],
        )
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
        self.ema = EMA(self.net, power=3 / 4, update_after_step=1, update_every=1, include_online_model=False)

        if self.train_on_latents:
            self.img_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device)
            self.img_vae.eval()
            self._set_requires_grad(self.img_vae, False)
            self.img_resolution, self.img_channels = self.img_resolution // 8, 4

    @torch.no_grad()
    def _encode_latents(self, images: torch.Tensor):
        """Encode the given images to compressed latent space.

        Args:
            images (`torch.Tensor`): The images to encode.

        Returns:
            The encoded latents.
        """
        images = self.img_vae.encode(images)["latent_dist"].sample()
        latents = self.latent_scale_factor * images

        return latents

    @torch.no_grad()
    def _decode_latents(self, samples: torch.Tensor):
        """Decode the given latents to images.

        Args:
            samples (`torch.Tensor`): The samples to decode.

        Returns:
            The decoded images.
        """
        latents = 1 / self.latent_scale_factor * samples
        samples = self.img_vae.decode(latents.float()).sample
        return samples

    def test(self, ckpt_list: list[str]):
        """Test the model given a list of checkpoints.

        Args:
            ckpt_list (list[str]): List of checkpoint paths
        """
        if os.path.exists(os.path.join(self.outdir, "test_metrics.csv")):
            os.remove(os.path.join(self.outdir, "test_metrics.csv"))

        if os.path.exists(os.path.join(self.outdir, "accuracy_per_class.csv")):
            os.remove(os.path.join(self.outdir, "accuracy_per_class.csv"))

        for ckpt in ckpt_list:
            self._load_checkpoint(ckpt)
            metrics = self.evaluate(self.ema, self.test_dataloader)
            ckpt_name = os.path.basename(ckpt).split(".")[0]
            self._save_metrics(ckpt_name, metrics)

    def get_features_and_labels(self, net: torch.nn.Module, ckpt: str):
        """Get features and labels for TSNE visualization given a checkpoint.

        Args:
            net (torch.nn.Module): Model to test.
            ckpt (str): Checkpoint path.

        Returns:
            np.ndarray: Intermediate features.
            np.ndarray: Labels.
        """
        features, labels, images_list, indices_list = self.get_intermediate_features(net, self.test_dataloader)
        return features, labels, images_list, indices_list

    def _save_metrics(self, ckpt_name: str, metrics: dict):
        """Save metrics to a CSV file.

        Args:
            ckpt_name (str): Checkpoint name.
            metrics (dict): Metrics to save.
        """
        csv_path = os.path.join(self.outdir, f"test_metrics-{self.ckpt_type}.csv")
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
        self.ema.eval()

    def _save_accuracy_per_class(self, acc_per_class: torch.Tensor):
        """Save accuracy per class to a CSV file.

        Args:
            acc_per_class (torch.Tensor): Accuracy per class.
        """
        csv_path = os.path.join(self.outdir, f"accuracy_per_class-{self.ckpt_type}.csv")
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([i for i in range(len(acc_per_class))])
            writer.writerow([acc.item() for acc in acc_per_class])


class EGCFeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = None

        def hook(module, input, output):
            # Average pooling over spatial dimensions to get fixed-size features
            if isinstance(output, tuple):
                output = output[0]
            self.features = output.mean(dim=(2, 3))  # Global average pooling

        self.model.out.register_forward_hook(hook)


class WRNFeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = None

        def hook(module, input, output):
            # Average pooling over spatial dimensions to get fixed-size features
            self.features = output.mean(dim=(2, 3))  # Global average pooling

        self.model.features.register_forward_hook(hook)


class WRNTester(Tester):
    """Tester for Wide Residual Networks."""

    def __init__(self, outdir: str, test_dataset_kwargs, network_kwargs, ckpt_type, train_on_latents=False):
        super().__init__(outdir, test_dataset_kwargs, network_kwargs, ckpt_type, train_on_latents)

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

        num_classes = self.label_dim
        correct = torch.zeros(num_classes, device=self.device)
        total = torch.zeros(num_classes, device=self.device)

        with tqdm(total=len(dataloader), desc="Testing", dynamic_ncols=True) as pbar:
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device).argmax(dim=1)

                if self.train_on_latents:
                    images = self._encode_latents(images)

                logits = net(images)
                preds = logits.argmax(dim=1)

                for c in range(num_classes):
                    class_mask = labels == c
                    total[c] += class_mask.sum()
                    correct[c] += (preds[class_mask] == c).sum()

                loss = torch.nn.functional.cross_entropy(logits, labels)
                acc = (logits.argmax(dim=1) == labels).float().mean()

                loss_meter.update(loss.mean(), images.size(0))
                acc_meter.update(acc, images.size(0))
                ece_meter.update(self.ece(logits, labels))
                pbar.update(1)

            metrics = {
                "test_cls_loss": loss_meter.compute().item(),
                "test_cls_acc": acc_meter.compute().item(),
                "test_cls_ece": ece_meter.compute().item(),
            }
            pbar.set_postfix(metrics)
            pbar.close()

        # Calculate accuracy per class, handling division by zero
        acc_per_class = torch.zeros_like(correct, dtype=torch.float32)
        for c in range(num_classes):
            if total[c] > 0:
                acc_per_class[c] = correct[c] / total[c]
            else:
                acc_per_class[c] = 0.0

        self._save_accuracy_per_class(acc_per_class)

        return metrics

    @torch.no_grad()
    def get_intermediate_features(self, ckpt: str):
        """Get intermediate features from the model.

        Args:
            ckpt (str): Checkpoint path.

        Returns:
            np.ndarray: Intermediate features.
            np.ndarray: Labels.
        """
        self._load_checkpoint(ckpt)
        self.ema.eval()
        feature_extractor = WRNFeatureExtractor(self.ema)

        features_list = []
        labels_list = []
        images_list = []
        indices_list = []
        current_idx = 0

        with tqdm(total=len(self.test_dataloader), desc="Getting Features", dynamic_ncols=True) as pbar:
            for images, labels in self.test_dataloader:
                images, labels = images.to(self.device), labels.to(self.device).argmax(dim=1)

                if self.train_on_latents:
                    images = self._encode_latents(images)

                images = get_patches(images, self.img_resolution)
                _ = self.ema(images)

                features = feature_extractor.features

                features_list.append(features.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

                if self.train_on_latents:
                    images = self._decode_latents(images)

                # Store original images and their indices
                images_list.extend(images.cpu())
                indices_list.extend(range(current_idx, current_idx + len(images)))
                current_idx += len(images)

                pbar.update(1)

        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        return features, labels, images_list, indices_list


class EGCTester(Tester):
    """Tester for EGC."""

    def __init__(self, outdir: str, test_dataset_kwargs, network_kwargs, ckpt_type, train_on_latents=False):
        super().__init__(outdir, test_dataset_kwargs, network_kwargs, ckpt_type, train_on_latents)

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
                images, labels = images.to(self.device), labels.to(self.device).argmax(dim=1)

                if self.train_on_latents:
                    images = self._encode_latents(images)

                images = get_patches(images, self.img_resolution)
                clean_timesteps = torch.zeros(images.size(0), dtype=torch.long, device=self.device)

                logits = net(images, clean_timesteps, cls_mode=True)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                acc = (logits.argmax(dim=1) == labels).float().mean()

                loss_meter.update(loss.mean(), images.size(0))
                acc_meter.update(acc, images.size(0))
                ece_meter.update(self.ece(logits, labels))
                pbar.update(1)

            metrics = {
                "test_cls_loss": loss_meter.compute().item(),
                "test_cls_acc": acc_meter.compute().item(),
                "test_cls_ece": ece_meter.compute().item(),
            }
            pbar.set_postfix(metrics)
            pbar.close()

        return metrics

    @torch.no_grad()
    def get_intermediate_features(self, ckpt: str):
        """Get intermediate features from the model.

        Args:
            ckpt (str): Checkpoint path.

        Returns:
            np.ndarray: Intermediate features.
            np.ndarray: Labels.
        """
        self._load_checkpoint(ckpt)
        self.ema.eval()
        feature_extractor = EGCFeatureExtractor(self.ema)

        features_list = []
        labels_list = []
        images_list = []
        indices_list = []
        current_idx = 0

        with tqdm(total=len(self.test_dataloader), desc="Getting Features", dynamic_ncols=True) as pbar:
            for images, labels in self.test_dataloader:
                images, labels = images.to(self.device), labels.to(self.device).argmax(dim=1)

                if self.train_on_latents:
                    images = self._encode_latents(images)

                images = get_patches(images, self.img_resolution)
                clean_timesteps = torch.zeros(images.size(0), dtype=torch.long, device=self.device)

                _ = self.ema(images, clean_timesteps, cls_mode=True)

                features = feature_extractor.features

                features_list.append(features.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

                num_channels = images.shape[1] - 2
                images = images[:, :num_channels]

                if self.train_on_latents:
                    images = self._decode_latents(images)

                # Store original images and their indices
                images_list.extend(images.cpu())
                indices_list.extend(range(current_idx, current_idx + len(images)))
                current_idx += len(images)

                pbar.update(1)

        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        return features, labels, images_list, indices_list
