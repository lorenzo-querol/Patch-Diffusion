import json
import os
from typing import OrderedDict

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, set_seed
from diffusers import AutoencoderKL
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from torch_uncertainty.post_processing import TemperatureScaler
from torchvision import transforms
from tqdm import tqdm

import dnnlib

from .ece import ECELoss
from .utils import Meter

accelerator = Accelerator()


class Trainer:
    def __init__(
        self,
        run_dir: str,
        dataset_kwargs,
        val_dataset_kwargs,
        test_dataset_kwargs,
        network_kwargs,
        optimizer_kwargs,
        batch_size: int = 128,
        accum_steps: int = 1,
        seed: int = 1,
        resume_from: str = None,
        train_on_latents: bool = False,
    ):
        """Common initialization parameters shared by all trainers.

        Args:
            run_dir (`str`): Output directory.
            dataset_kwargs (`dict`): Training dataset options.
            val_dataset_kwargs (`dict`): Validation dataset options.
            test_dataset_kwargs (`dict`): Test dataset options.
            network_kwargs (`dict`): Model options.
            optimizer_kwargs (`dict`): Optimizer options.
            batch_size (`int`): Batch size.
            accum_steps (`int`): Gradient accumulation steps.
            seed (`int`): Random seed.
            resume_from (`str`): Path to resume checkpoint from.
            train_on_latents (`bool`): Whether to train on VAE latents.
        """
        self.run_dir = run_dir
        self.dataset_kwargs = dataset_kwargs
        self.val_dataset_kwargs = val_dataset_kwargs
        self.test_dataset_kwargs = test_dataset_kwargs
        self.network_kwargs = network_kwargs
        self.optimizer_kwargs = optimizer_kwargs

        self.batch_size = batch_size
        self.accum_steps = accum_steps
        self.seed = seed
        self.resume_from = resume_from
        self.train_on_latents = train_on_latents

        self.ece = ECELoss(n_bins=10)

        self.accelerator = Accelerator(
            dataloader_config=DataLoaderConfiguration(dispatch_batches=True, split_batches=False),
            gradient_accumulation_steps=self.accum_steps,
        )
        self.device = self.accelerator.device
        self.print_fn = self.accelerator.print

        self.per_device_batch_size = self._calculate_per_device_batch_size()
        self.dataloader_kwargs = dict(
            batch_size=self.per_device_batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.img_vae = None
        self.latent_scale_factor = 0.18215

    def _init_env(self):
        """Sets seeds and other environment variables."""
        set_seed(self.seed)
        torch.backends.cudnn.benchmark = True

    def _calculate_per_device_batch_size(self):
        """Calculate the batch size per device."""
        world_size = self.accelerator.num_processes
        per_device_batch_size = self.batch_size // (world_size * self.accelerator.gradient_accumulation_steps)
        assert per_device_batch_size * world_size * self.accelerator.gradient_accumulation_steps == self.batch_size, "Batch size must be divisible by num_processes * gradient_accumulation_steps."
        return per_device_batch_size

    def _prepare_datasets(self):
        """Prepare datasets."""
        self.print_fn("Loading datasets...")

        dataset_obj = dnnlib.util.construct_class_by_name(**self.dataset_kwargs)
        self.img_resolution, self.img_channels, self.label_dim = (
            dataset_obj.resolution,
            dataset_obj.num_channels,
            dataset_obj.label_dim,
        )
        del dataset_obj

        augment_transform = transforms.Compose(
            [
                transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomCrop(self.img_resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * self.img_channels, std=[0.5] * self.img_channels),
            ]
        )
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * self.img_channels, std=[0.5] * self.img_channels),
            ]
        )

        self.cls_dataset = dnnlib.util.construct_class_by_name(**self.dataset_kwargs, transform=augment_transform)
        self.val_dataset = dnnlib.util.construct_class_by_name(**self.val_dataset_kwargs, transform=transform)
        self.test_dataset = dnnlib.util.construct_class_by_name(**self.test_dataset_kwargs, transform=transform)

    def _encode_latents(self, images: torch.Tensor):
        """Encode the given images to compressed latent space.

        Args:
            images (`torch.Tensor`): The images to encode.

        Returns:
            The encoded latents.
        """
        with torch.no_grad():
            images = self.img_vae.encode(images)["latent_dist"].sample()
            latents = self.latent_scale_factor * images

        return latents

    def _set_requires_grad(self, model: torch.nn.Module, requires_grad: bool):
        """Set requires_grad for all parameters in the model.

        Args:
            model (torch.nn.Module): The model to set `requires_grad` for.
            requires_grad (bool): Whether to set `requires_grad` to true or false.
        """
        for param in model.parameters():
            param.requires_grad = requires_grad

    @accelerator.on_main_process
    def _update_ema(self):
        """Update the EMA model."""
        self.ema.to(self.device)
        self.ema.update()

    @accelerator.on_main_process
    def _save_checkpoint(self, filename: str):
        """Save a checkpoint file.

        Args:
            filename (`str`): The name of the file to save.
        """
        data = {"ema": self.ema.state_dict()}
        self.print_fn(f"Saving checkpoint to {filename}...")
        torch.save(data, os.path.join(self.run_dir, f"{filename}.pt"))

    def _load_checkpoint(self):
        """Load the latest checkpoint specified by `self.resume_from`.\n

        If `self.resume_from` is a directory, then the latest checkpoint is loaded from that directory.\n
        If `self.resume_from` is a file, then the checkpoint is loaded from that file.\n
        If `self.resume_from` is `None`, then no checkpoint is loaded.
        """
        if self.resume_from is None:
            return

        ckpt = self.resume_from
        if self.resume_from.endswith(".pt"):
            pt_files = [f for f in os.listdir(self.resume_from) if f.endswith(".pt")]
            ckpt = os.path.join(self.resume_from, pt_files[-1])

        self.print_fn(f"Resuming from {ckpt}...")
        data = torch.load(ckpt, weights_only=True)

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])
            self.ema.to(self.device)

    def _compute_norms(self):
        """Compute the gradient and parameter norms."""
        grad_norm = 0.0
        for p in self.net.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm(2) ** 2
        grad_norm = grad_norm**0.5

        param_norm = 0.0
        for p in self.net.parameters():
            param_norm += p.norm(2) ** 2
        param_norm = param_norm**0.5

        return grad_norm, param_norm


class WRNTrainer(Trainer):
    """Trainer for Wide Residual Network."""

    def __init__(
        self,
        run_dir: str,
        dataset_kwargs,
        val_dataset_kwargs,
        test_dataset_kwargs,
        network_kwargs,
        optimizer_kwargs,
        decay_epochs: list[int] = [60, 120, 160],
        decay_rate: float = 0.2,
        num_epochs: int = 200,
        accum_steps: int = 1,
        batch_size: int = 128,
        seed: int = 1,
        resume_from: str = None,
        calibrate: bool = False,
        train_on_latents: bool = False,
    ):
        """WRN-specific initialization.

        Args:
            decay_epochs (`list[int]`): When to decay learning rate.
            decay_rate (`float`): Learning rate decay factor.
            num_epochs (`int`): Total training epochs.
            calibrate (`bool`): Whether to calibrate model temperatures.
        """
        super().__init__(
            run_dir=run_dir,
            dataset_kwargs=dataset_kwargs,
            val_dataset_kwargs=val_dataset_kwargs,
            test_dataset_kwargs=test_dataset_kwargs,
            network_kwargs=network_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            batch_size=batch_size,
            accum_steps=accum_steps,
            seed=seed,
            resume_from=resume_from,
            train_on_latents=train_on_latents,
        )

        self.decay_epochs = decay_epochs
        self.decay_rate = decay_rate
        self.num_epochs = num_epochs
        self.calibrate = calibrate

        self._init_trainer()

    def _init_trainer(self):
        self._init_env()
        self._prepare_datasets()
        self._prepare_dataloaders()
        self._build_network()

        if self.resume_from is not None:
            self._load_checkpoint()

    def _prepare_dataloaders(self):
        """Prepare dataloaders."""
        self.cls_dataloader = DataLoader(self.cls_dataset, **self.dataloader_kwargs)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=4)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=4)
        self.cls_dataloader, self.val_dataloader, self.test_dataloader = self.accelerator.prepare(self.cls_dataloader, self.val_dataloader, self.test_dataloader)

    def _build_network(self):
        """Setup network."""
        self.print_fn("Setting up network...")

        if self.train_on_latents:
            self.img_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device)
            self.img_vae.eval()
            self._set_requires_grad(self.img_vae, False)
            self.img_resolution, self.img_channels = self.img_resolution // 8, 4

        self.network_kwargs.update({"label_dim": self.label_dim, "in_channels": self.img_channels})
        self.net = dnnlib.util.construct_class_by_name(**self.network_kwargs)

        if self.accelerator.is_main_process:
            with open(os.path.join(self.run_dir, "network_kwargs.json"), "w") as f:
                json.dump(self.network_kwargs, f, indent=4)

        # Setup the EMA
        if self.accelerator.is_main_process:
            self.ema = EMA(self.net, power=3 / 4, include_online_model=False)

        # Setup the optimizer
        self.print_fn("Setting up optimizer...")
        self.optimizer = dnnlib.util.construct_class_by_name(params=self.net.parameters(), **self.optimizer_kwargs)

        # Prepare for distributed training
        self.net, self.optimizer = self.accelerator.prepare(self.net, self.optimizer)

    def train(self, eval_interval: int):
        """
        Main training loop.

        Args:
            eval_interval (`int`): When to evaluate the model.
        """
        self.accelerator.init_trackers(project_name="EGC")

        for epoch in range(self.num_epochs):
            self.cur_epoch = epoch
            self._train_one_epoch()

            if eval_interval > 0 and self.cur_epoch % eval_interval == 0:
                model = self.net

                if self.calibrate:
                    self.print_fn("Calibrating the model...")
                    model = TemperatureScaler(model=self.base_trainer.net, device=self.base_trainer.device)
                    model.fit(calibration_set=self.base_trainer.val_dataset)

                self.evaluate(model, self.test_dataloader)

        self.cur_epoch = self.num_epochs
        self.evaluate(self.net, self.test_dataloader)
        self._save_checkpoint("model-final")

    def _gather(self, metrics: OrderedDict):
        """Gather the metrics across all processes."""
        global_metrics = self.accelerator.gather_for_metrics(metrics)
        if self.accelerator.is_local_main_process:
            global_metrics = {k: v.mean().item() for k, v in global_metrics.items()}
        return global_metrics

    def _train_one_epoch(self):
        """Train the model for one epoch."""
        cls_loss_meter = Meter()
        cls_acc_meter = Meter()
        cls_ece_meter = Meter()

        self.net.train()

        with tqdm(total=len(self.cls_dataloader), desc=f"Epoch {self.cur_epoch}", disable=not self.accelerator.is_local_main_process, dynamic_ncols=True) as pbar:
            for images, labels in self.cls_dataloader:
                pbar.update(1)

                labels = labels.argmax(dim=1)

                if self.train_on_latents:
                    images = self._encode_latents(images)

                logits = self.net(images)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                acc = (logits.argmax(dim=1) == labels).float().mean()
                ece = self.ece(logits, labels)

                cls_loss_meter.update(loss.mean(), images.size(0))
                cls_acc_meter.update(acc, images.size(0))
                cls_ece_meter.update(ece, images.size(0))

                self.accelerator.backward(loss)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self._update_ema()

            metrics = {
                "cls_loss": cls_loss_meter.compute().clone().detach(),
                "cls_acc": cls_acc_meter.compute().clone().detach(),
                "cls_ece": cls_ece_meter.compute().clone().detach(),
                "lr": torch.tensor(self.optimizer.param_groups[0]["lr"], device=self.device),
            }
            metrics = self._gather(metrics)
            pbar.set_postfix(metrics)
            pbar.close()

        self.accelerator.log(metrics, step=self.cur_epoch)

    @torch.no_grad()
    def evaluate(self, net: torch.nn.Module, dataloader: DataLoader):
        """Evaluate the model on the given dataloader.

        Args:
            net (`torch.nn.Module`): The model to evaluate.
            dataloader (`DataLoader`): A validation dataloader.

        Returns:
            The computed metrics.
        """
        dataloader = self.accelerator.prepare(dataloader)
        loss_meter = Meter()
        acc_meter = Meter()
        ece_meter = Meter()

        net.eval()

        with tqdm(total=len(dataloader), desc="Validation", disable=not self.accelerator.is_local_main_process, dynamic_ncols=True) as pbar:
            for images, labels in dataloader:
                pbar.update(1)

                labels = labels.argmax(dim=1)

                if self.train_on_latents:
                    images = self._encode_latents(images)

                logits = net(images)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                acc = (logits.argmax(dim=1) == labels).float().mean()
                ece = self.ece(logits, labels)

                loss_meter.update(loss.mean(), images.size(0))
                acc_meter.update(acc, images.size(0))
                ece_meter.update(ece, images.size(0))

            metrics = {
                "val_cls_loss": loss_meter.compute().clone().detach(),
                "val_cls_acc": acc_meter.compute().clone().detach(),
                "val_cls_ece": ece_meter.compute().clone().detach(),
            }
            metrics = self._gather(metrics)
            pbar.set_postfix(metrics)
            pbar.close()

        self.accelerator.log(metrics, step=self.cur_epoch)

    @torch.no_grad()
    def get_probs(self, net: torch.nn.Module, dataloader: DataLoader):
        """Get the computed probabilities.

        Args:
            net (`torch.nn.Module`): The model to use for computing probabilities.
            dataloader (`DataLoader`): The dataloader to use for computing probabilities.

        Returns:
            The computed probabilities.
        """
        dataloader = self.accelerator.prepare(dataloader)
        probs = []

        net.eval()

        with tqdm(total=len(dataloader), desc="Computing Probabilities", disable=not self.accelerator.is_local_main_process, dynamic_ncols=True) as pbar:
            for images, _ in dataloader:
                pbar.update(1)

                if self.train_on_latents:
                    images = self._encode_latents(images)

                logits = net(images)
                prob = logits.softmax(dim=1)
                probs.append(prob)

            pbar.close()

        net.train()

        return torch.cat(probs, dim=0)
