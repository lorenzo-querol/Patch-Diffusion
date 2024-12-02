import copy
import json
import math
import os
import pickle
import time

import numpy as np
import torch
from diffusers import AutoencoderKL
import torchvision
from tqdm import tqdm
from calibration.ece import ECE
from training.resample import UniformSampler
from .patch import get_patches
from .utils import set_requires_grad, cycle
from torch.utils.data import DataLoader
import dnnlib
from torch_utils import distributed as dist
from torch_utils import misc, training_stats
from torchvision import transforms
from accelerate import Accelerator
from ema_pytorch import EMA
from accelerate.utils import set_seed
from .ece import ECELoss
from torch.optim.lr_scheduler import OneCycleLR


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def get_patch_size_list(img_resolution, real_p, train_on_latents):
    is_smaller_than_64 = img_resolution == 32

    if is_smaller_than_64:
        batch_mul_dict = {32: 1, 16: 4}  # Simplified multipliers for 32x32
        if real_p < 1.0:
            p_list = np.array([(1 - real_p), real_p])
            patch_list = np.array([16, 32])  # Only two patch sizes for 32x32
            batch_mul_avg = np.sum(p_list * np.array([4, 1]))
        else:
            p_list = np.array([0, 1.0])
            patch_list = np.array([16, 32])
            batch_mul_avg = 1
    else:
        batch_mul_dict = {512: 1, 256: 2, 128: 4, 64: 16, 32: 32, 16: 64}
        if train_on_latents:
            p_list = np.array([(1 - real_p), real_p])
            patch_list = np.array([img_resolution // 2, img_resolution])
            batch_mul_avg = np.sum(p_list * np.array([2, 1]))
        else:
            p_list = np.array([(1 - real_p) * 2 / 5, (1 - real_p) * 3 / 5, real_p])
            patch_list = np.array([img_resolution // 4, img_resolution // 2, img_resolution])
            batch_mul_avg = np.sum(np.array(p_list) * np.array([4, 2, 1]))  # 2

    return p_list, patch_list, batch_mul_dict, batch_mul_avg


def encode_images_to_latents(img_vae, images, latent_scale_factor: float, train_on_latents: bool):
    """Encode images to latents using the given VAE. Return latents if train_on_latents is True, otherwise the input images.
    Args:
        img_vae: VAE model.
        images: Input images.
        latent_scale_factor (float): Scaling factor for latents.
        train_on_latents (bool): Whether to train on latents.
    Returns:
        Latents if train_on_latents is True, otherwise the input images.
    """
    if train_on_latents:
        assert img_vae is not None, "img_vae must be provided when train_on_latents is True."
        with torch.no_grad():
            images = img_vae.encode(images)["latent_dist"].sample()
            images = latent_scale_factor * images
    return images


def get_batch_data(dataset_iterator, device, batch_mul=1):
    """Get a batch of data from the dataset iterator.

    :param dataset_iterator: The dataset iterator.
    :param device: The device to move the data to.
    :param batch_mul: The number of batches to get from the iterator. Default is 1.
    :return: A tuple of images and labels.
    """
    images, labels = [], []

    for _ in range(batch_mul):
        images_, labels_ = next(dataset_iterator)
        images.append(images_), labels.append(labels_)
    images, labels = torch.cat(images).to(device), torch.cat(labels).to(device)
    return images, labels


class Trainer:
    def __init__(
        self,
        run_dir="./training-runs",  # Output directory
        dataset_kwargs={},  # Training dataset options
        val_dataset_kwargs={},  # Validation dataset options
        network_kwargs={},  # Model options
        diffusion_kwargs={},  # Diffusion options
        optimizer_kwargs={},  # Optimizer options
        num_steps=100_000,  # Number of training steps
        batch_size=128,  # Batch size
        ce_weight=0.001,  # Weight of the classification loss
        label_smooth=0.2,  # Label smoothing factor
        real_p=0.5,  # Probability of real images
        train_on_latents=False,  # Train on latent representations
        seed=1,  # Seed for reproducibility
        resume_from=None,  # Checkpoint to resume from
    ):
        self.run_dir = run_dir
        self.dataset_kwargs = dataset_kwargs
        self.val_dataset_kwargs = val_dataset_kwargs
        self.network_kwargs = network_kwargs
        self.diffusion_kwargs = diffusion_kwargs
        self.optimizer_kwargs = optimizer_kwargs

        self.num_steps = num_steps
        self.ce_weight = ce_weight
        self.label_smooth = label_smooth
        self.real_p = real_p
        self.train_on_latents = train_on_latents
        self.seed = seed
        self.resume_from = resume_from

        self.batch_size = batch_size

        self.criterion = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=self.label_smooth)
        self.ece_criterion = ECELoss(n_bins=10)

        self.cur_step = 0
        self.accelerator = Accelerator(split_batches=True, log_with="wandb", gradient_accumulation_steps=2)
        self.device = self.accelerator.device
        self.print_fn = self.accelerator.print
        self._init_trainer()

    def _init_trainer(self):
        """Initialize the Trainer: seeds, datasets, and network."""
        self._init_env()
        self._prepare_dataloaders()
        self._build_network_and_diffusion()

        if self.resume_from is not None:
            self._load()

    def _init_env(self):
        set_seed(self.seed)
        torch.backends.cudnn.benchmark = True

    def _prepare_dataloaders(self):
        """Prepare datasets and dataloaders."""
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
                transforms.RandomResizedCrop(self.img_resolution),
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
        dataloader_kwargs = dict(
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        self.train_dataloader = DataLoader(self.train_dataset, **dataloader_kwargs)
        self.cls_dataloader = DataLoader(self.cls_dataset, **dataloader_kwargs)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=4)

        self.train_dataloader, self.cls_dataloader, self.val_dataloader = self.accelerator.prepare(
            self.train_dataloader,
            self.cls_dataloader,
            self.val_dataloader,
        )

        self.train_dataloader, self.cls_dataloader = cycle(self.train_dataloader), cycle(self.cls_dataloader)

    def _set_requires_grad(self, model, requires_grad):
        """Set requires_grad for all parameters in the model"""
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _build_network_and_diffusion(self):
        """Setup network and diffusion"""
        self.print_fn("Constructing network and diffusion...")
        self.net = dnnlib.util.construct_class_by_name(
            **self.network_kwargs,
            img_resolution=self.img_resolution,
            in_channels=self.img_channels,
            out_channels=self.label_dim,
            label_dim=self.label_dim,
        )
        self.diffusion = dnnlib.util.construct_class_by_name(**self.diffusion_kwargs)
        self.schedule_sampler = UniformSampler(self.diffusion)

        # ---------------------------------------------------------------------
        """ Setup EMA """
        self.print_fn("Setting up EMA...")
        if self.accelerator.is_main_process:
            self.ema = EMA(self.net)

        # ---------------------------------------------------------------------
        """ Setup the optimizer and scheduler """
        self.print_fn("Setting up optimizer and scheduler...")
        self.optimizer = dnnlib.util.construct_class_by_name(params=self.net.parameters(), **self.optimizer_kwargs)

        # ---------------------------------------------------------------------
        """ Prepare for distributed training """
        self.net, self.optimizer = self.accelerator.prepare(
            self.net,
            self.optimizer,
        )

        # BUG: Acclerate's prepare resets the model to NOT require gradients, so we need to set it back to True!!
        self._set_requires_grad(self.net, True)

    def train(self, print_interval=10, eval_interval=100, save_interval=1000):
        """Main training loop.

        :param num_steps: Total number of training steps.
        :param print_interval: How often to print metrics. Default is 10 steps.
        :param eval_interval: When to evaluate the model. Default is 100 steps.
        :param save_interval: When to save the model. Default is 1000 steps.
        """

        self.print_fn(f"Training for {self.num_steps - self.cur_step} steps...")
        self.print_fn("")

        self.accelerator.init_trackers(project_name="EGC")

        while self.cur_step < self.num_steps:
            metrics = self._training_step()
            self._update_ema()

            if self.cur_step % print_interval == 0:
                self.print_fn()
                self.print_fn(f"Step {self.cur_step}/{self.num_steps}")
                self._report_metrics(metrics)

            if self.ce_weight > 0 and self.cur_step % eval_interval == 0:
                metrics = self._evaluate()
                self._report_metrics(metrics)

            if self.cur_step % save_interval == 0:
                self._save()
                self._sample_images(self.img_resolution, self.img_channels)

            self.cur_step += 1

    def _update_ema(self):
        if not self.accelerator.is_main_process:
            return

        self.ema.to(self.device)
        self.ema.update()

    def _sample_images(self, img_size, img_channels, num_images=16):
        if not self.accelerator.is_main_process:
            return

        self.ema.eval()

        samples = torch.randn((num_images, img_channels, img_size, img_size), device=self.device)

        for t in reversed(range(0, self.diffusion.num_timesteps)):
            tt = torch.full((num_images,), t, device=self.device, dtype=torch.long)

            with torch.no_grad():
                pred_noise = self.ema(samples, tt)
                samples = self.diffusion.p_sample(pred_noise, samples, tt)

        image_grid = torchvision.utils.make_grid(samples, nrow=int(math.sqrt(num_images)), normalize=True, scale_each=True)
        torchvision.utils.save_image(image_grid, os.path.join(self.run_dir, f"sample-{self.cur_step}.png"))

        return samples

    def _training_step(self):
        loss = 0
        metrics = {}

        self.optimizer.zero_grad(set_to_none=True)

        # ---------------------------------------------------------------------
        """Classification Component"""

        if self.ce_weight > 0:
            cls_images, cls_labels = next(self.cls_dataloader)
            clean_images = cls_images

            t_noised, _ = self.schedule_sampler.sample(cls_images.shape[0], self.device)
            t_clean = torch.zeros(cls_images.shape[0], dtype=torch.long, device=self.device)

            images_all = torch.cat([cls_images, clean_images])
            timesteps_all = torch.cat([t_noised, t_clean])
            sqrt_alphas_cumprod = torch.from_numpy(self.diffusion.sqrt_alphas_cumprod).to(self.device)[timesteps_all].float()

            logits = self.net(images_all, timesteps_all, cls_mode=True)

            cls_labels = torch.cat([cls_labels, cls_labels]).argmax(dim=1)
            ce_loss = self.criterion(logits, cls_labels)
            cls_loss = (ce_loss * sqrt_alphas_cumprod).mean()
            cls_acc = (logits.argmax(dim=1) == cls_labels).float().mean()
            cls_ece = self.ece_criterion(logits, cls_labels)

            metrics["cls_loss"] = cls_loss
            metrics["cls_acc"] = cls_acc
            metrics["cls_ece"] = cls_ece

            loss += self.ce_weight * cls_loss

        # ---------------------------------------------------------------------
        """ Diffusion Component """

        images, labels = next(self.train_dataloader)
        timesteps, weights = self.schedule_sampler.sample(images.shape[0], self.device)

        mask = torch.rand(labels.shape[0], device=self.device) < 0.1
        labels[mask] = self.label_dim

        mse_loss = self.diffusion.training_losses(
            net=self.net,
            x_start=images,
            t=timesteps,
            model_kwargs={"class_labels": labels.argmax(dim=1)},
        )

        mse_loss = (mse_loss * weights).mean()
        metrics["mse_loss"] = mse_loss

        loss += mse_loss

        # ---------------------------------------------------------------------
        """ Backward pass and optimizer step """

        self.accelerator.backward(loss)
        self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)

        self.accelerator.wait_for_everyone()
        self.optimizer.step()
        self.accelerator.wait_for_everyone()

        # ---------------------------------------------------------------------
        """ Compute gradients """

        grad_norm = 0.0
        for p in self.net.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm(2) ** 2
        grad_norm = grad_norm**0.5

        param_norm = 0.0
        for p in self.net.parameters():
            param_norm += p.norm(2) ** 2
        param_norm = param_norm**0.5

        metrics["grad_norm"] = grad_norm
        metrics["param_norm"] = param_norm

        return metrics

    def _evaluate(self):
        metrics = {}
        all_losses, all_accs, all_eces = [], [], []

        self.net.eval()

        with torch.no_grad():
            for images, labels in self.val_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                timesteps = torch.zeros(images.shape[0], dtype=torch.long, device=self.device)

                logits = self.net(images, timesteps, cls_mode=True)

                labels = labels.argmax(dim=1)
                ce_loss = torch.nn.functional.cross_entropy(logits, labels)
                acc = (logits.argmax(dim=1) == labels).float().mean()
                ece = self.ece_criterion(logits, labels)

                all_losses.append(ce_loss)
                all_accs.append(acc)
                all_eces.append(ece)

        self.net.train()

        all_losses = torch.stack(all_losses)
        all_accs = torch.stack(all_accs)
        all_eces = torch.stack(all_eces)

        metrics["val_cls_loss"] = all_losses.mean()
        metrics["val_cls_acc"] = all_accs.mean()
        metrics["val_cls_ece"] = all_eces.mean()

        return metrics

    def _report_metrics(self, metrics: dict):
        """
        Gathers and reports metrics across all processes using Accelerate.

        Args:
            metrics (dict): Dictionary of metrics with names as keys and tensors as values.
                            Example: {"loss": tensor_loss, "accuracy": tensor_acc}.
        Returns:
            dict: A dictionary with global (averaged) metric values.
        """
        global_metrics = {}
        for name, value in metrics.items():
            gathered_values = self.accelerator.gather_for_metrics(value)
            if self.accelerator.is_main_process:
                global_avg = gathered_values.mean().item()
                global_metrics[name] = global_avg
                self.print_fn(f"{name} = {global_avg:.6f}")

        self.accelerator.log(global_metrics, step=self.cur_step)

    def _save(self):
        if not self.accelerator.is_main_process:
            return

        data = {
            "step": self.cur_step,
            "net": self.accelerator.unwrap_model(self.net).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
        }

        torch.save(data, os.path.join(self.run_dir, f"model-{self.cur_step:06d}.pt"))

    def _load(self):
        pt_files = [f for f in os.listdir(self.resume_from) if f.endswith(".pt")]
        latest_ckpt = os.path.join(self.resume_from, pt_files[-1])

        data = torch.load(latest_ckpt, map_location=self.device, weights_only=True)

        self.accelerator.unwrap_model(self.net).load_state_dict(data["net"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.ema.load_state_dict(data["ema"])

        if not all(param.requires_grad for param in self.net.parameters()):
            self.print_fn("Some parameters in the model are not trainable. Setting all parameters to trainable...")
            self._set_requires_grad(self.net, True)

        self.cur_step = data["step"]
