import os
from typing import OrderedDict

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, set_seed
from torchvision import transforms

import dnnlib
from training.ece import ECELoss

accelerator = Accelerator()


class BaseTrainer:
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
        self.al_mul = 0

        self.ece = ECELoss(n_bins=10)

        self.accelerator = Accelerator(
            dataloader_config=DataLoaderConfiguration(dispatch_batches=True, split_batches=False),
            log_with="wandb",
            gradient_accumulation_steps=self.accum_steps,
        )
        self.accelerator.init_trackers(project_name="EGC")
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

        self.train_dataset = dnnlib.util.construct_class_by_name(**self.dataset_kwargs, transform=transform)
        self.cls_dataset = dnnlib.util.construct_class_by_name(**self.dataset_kwargs, transform=augment_transform)
        self.val_dataset = dnnlib.util.construct_class_by_name(**self.val_dataset_kwargs, transform=transform)
        self.test_dataset = dnnlib.util.construct_class_by_name(**self.test_dataset_kwargs, transform=transform)

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

    def _gather(self, metrics: OrderedDict):
        """Gather the metrics across all processes."""
        global_metrics = self.accelerator.gather_for_metrics(metrics)
        if self.accelerator.is_local_main_process:
            global_metrics = {k: v.mean().item() for k, v in global_metrics.items()}
        return global_metrics
