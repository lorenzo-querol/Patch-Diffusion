import copy
import json
import math
import os
from typing import OrderedDict

import numpy as np
import torch
from torchinfo import summary
import torchvision
from accelerate import Accelerator
from accelerate.utils import set_seed, DataLoaderConfiguration
from diffusers import AutoencoderKL
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import dnnlib

from .ece import ECELoss
from .patch import get_patches
from .utils import Meter, cycle

accelerator = Accelerator()


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
        test_dataset_kwargs={},  # Test dataset options
        network_kwargs={},  # Model options
        diffusion_kwargs={},  # Diffusion options
        optimizer_kwargs={},  # Optimizer options
        num_epochs=200,  # Number of training steps
        accum_steps=1,  # Accumulate gradients over multiple steps
        batch_size=128,  # Batch size
        ce_weight=0.001,  # Weight of the classification loss
        real_p=0.5,  # Probability of real images
        target="epsilon",  # Target for the diffusion model
        train_on_latents=False,  # Train on latent representations
        seed=1,  # Seed for reproducibility
        resume_from=None,  # Checkpoint to resume from
    ):
        self.run_dir = run_dir
        self.dataset_kwargs = dataset_kwargs
        self.val_dataset_kwargs = val_dataset_kwargs
        self.test_dataset_kwargs = test_dataset_kwargs
        self.network_kwargs = network_kwargs
        self.diffusion_kwargs = diffusion_kwargs
        self.optimizer_kwargs = optimizer_kwargs

        self.num_epochs = num_epochs
        self.accum_steps = accum_steps
        self.ce_weight = ce_weight
        self.real_p = real_p
        self.target = target
        self.train_on_latents = train_on_latents
        self.seed = seed
        self.resume_from = resume_from
        self.batch_size = batch_size

        self.ece = ECELoss(n_bins=10)

        self.accelerator = Accelerator(
            dataloader_config=DataLoaderConfiguration(dispatch_batches=True, split_batches=False),
            gradient_accumulation_steps=self.accum_steps,
        )
        self.device = self.accelerator.device
        self.print_fn = self.accelerator.print
        self.per_device_batch_size = self._calculate_per_device_batch_size()
        self._init_trainer()

    def _calculate_per_device_batch_size(self):
        """Calculate the batch size per device."""
        world_size = self.accelerator.num_processes
        per_device_batch_size = self.batch_size // (world_size * self.accelerator.gradient_accumulation_steps)
        assert per_device_batch_size * world_size * self.accelerator.gradient_accumulation_steps == self.batch_size, "Batch size must be divisible by num_processes * gradient_accumulation_steps."
        return per_device_batch_size

    def _init_trainer(self):
        """Initialize the Trainer: seeds, datasets, and network."""
        self._init_env()
        self._prepare_dataloaders()
        self._build_network()
        self._build_optimizer()
        self._build_ema()
        self._build_sampler()
        self._prepare_patch_info()

        if self.resume_from is not None:
            self._load()

        # NOTE: We need to initialize the diffusion model after loading the checkpoint if it exists
        self.diffusion = dnnlib.util.construct_class_by_name(**self.diffusion_kwargs, model=self.net)
        self.diffusion = self.accelerator.prepare(self.diffusion)

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
        self.dataloader_kwargs = dict(
            batch_size=self.per_device_batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
            generator=torch.Generator().manual_seed(self.seed),
        )
        self.train_dataloader = DataLoader(self.train_dataset, **self.dataloader_kwargs)
        self.cls_dataloader = DataLoader(self.cls_dataset, **self.dataloader_kwargs)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=4)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=4)

        self.train_dataloader, self.cls_dataloader, self.val_dataloader, self.test_dataloader = self.accelerator.prepare(
            self.train_dataloader,
            self.cls_dataloader,
            self.val_dataloader,
            self.test_dataloader,
        )

        self.cls_dataloader = cycle(self.cls_dataloader)

    def _set_requires_grad(self, model, requires_grad):
        """Set requires_grad for all parameters in the model."""
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _prepare_patch_info(self):
        """Prepare the patch sizes and probabilities."""
        real_p = self.real_p
        img_resolution = self.img_resolution
        is_32 = self.img_resolution == 32

        if is_32:
            batch_mul_dict = {32: 1, 16: 4}  # Simplified multipliers for 32x32
            if self.real_p < 1.0:
                p_list = np.array([(1 - real_p), real_p])
                patch_list = np.array([16, 32])  # Only two patch sizes for 32x32
            else:
                p_list = np.array([0, 1.0])
                patch_list = np.array([16, 32])
        else:
            """Default options for Patch Diffusion"""
            batch_mul_dict = {512: 1, 256: 2, 128: 4, 64: 16, 32: 32, 16: 64}

            if self.train_on_latents:
                p_list = np.array([(1 - real_p), real_p])
                patch_list = np.array([img_resolution // 2, img_resolution])
            else:
                p_list = np.array([(1 - real_p) * 2 / 5, (1 - real_p) * 3 / 5, real_p])
                patch_list = np.array([img_resolution // 4, img_resolution // 2, img_resolution])

        self.p_list = p_list
        self.patch_list = patch_list
        self.batch_mul_dict = batch_mul_dict

    def _build_network(self):
        """Build the EBM-UNet."""
        self.print_fn("Setting up network...")

        attention_ds = []
        for res in self.network_kwargs["attn_resolutions"]:
            attention_ds.append(self.img_resolution // int(res))

        self.network_kwargs.update({"attn_resolutions": tuple(attention_ds)})

        self.img_vae = None
        if self.train_on_latents:
            self.img_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device)
            self.img_vae.eval()
            self._set_requires_grad(self.img_vae, False)
            self.latent_scale_factor = 0.18215
            self.img_resolution, self.img_channels = self.img_resolution // 8, 4

        self.network_kwargs.update({"img_resolution": self.img_resolution, "in_channels": self.img_channels + 2, "out_channels": self.label_dim, "label_dim": self.label_dim})
        self.net = dnnlib.util.construct_class_by_name(**self.network_kwargs)

        if self.accelerator.is_local_main_process:
            with open(os.path.join(self.run_dir, "network_kwargs.json"), "w") as f:
                json.dump(self.network_kwargs, f, indent=4)

        self.net = self.accelerator.prepare(self.net)
        self._set_requires_grad(self.net, True)  # NOTE (BUG): Accelerate's prepare method for some reason resets the model to NOT require gradients

        summary(
            self.net,
            input_size=(self.per_device_batch_size, self.img_channels, self.img_resolution, self.img_resolution),
            col_names=("input_size", "output_size", "num_params", "trainable"),
            device="cuda",
        )

    def _build_optimizer(self):
        self.print_fn("Setting up optimizer...")
        self.optimizer = dnnlib.util.construct_class_by_name(params=self.net.parameters(), **self.optimizer_kwargs)
        self.optimizer = self.accelerator.prepare(self.optimizer)

    @accelerator.on_local_main_process
    def _build_ema(self):
        self.print_fn("Setting up EMA...")
        self.ema = EMA()
        self.ema = self.ema.to(self.device)

    @accelerator.on_local_main_process
    def _build_sampler(self):
        """Build the DDIM sampler."""
        self.print_fn("Setting up DDIM sampler...")
        sampler_kwargs = copy.deepcopy(self.diffusion_kwargs)
        sampler_kwargs.update({"class_name": "training.diffusion.DDIMSampler"})

        with open(os.path.join(self.run_dir, "sampler_kwargs.json"), "w") as f:
            json.dump(self.network_kwargs, f, indent=4)

        self.sampler = dnnlib.util.construct_class_by_name(**sampler_kwargs, model=self.ema)

    @accelerator.on_local_main_process
    def _sample_images(self, num_images=25):
        """Sample images from the model."""
        self.ema.eval()

        x_start, y_start, resolution, image_size = (0, 0, self.img_resolution, self.img_resolution)
        x_pos = torch.arange(x_start, x_start + image_size).view(1, -1).repeat(image_size, 1)
        y_pos = torch.arange(y_start, y_start + image_size).view(-1, 1).repeat(1, image_size)
        x_pos = (x_pos / (resolution - 1) - 0.5) * 2.0
        y_pos = (y_pos / (resolution - 1) - 0.5) * 2.0

        pos = torch.stack([x_pos, y_pos], dim=0).to(self.device)
        pos = pos.unsqueeze(0).repeat(num_images, 1, 1, 1)
        shape = (num_images, self.img_channels, self.img_resolution, self.img_resolution)
        x_0 = torch.randn(shape, device=self.device)
        class_labels = torch.randint(0, self.label_dim, (num_images,), device=self.device)

        samples = self.sampler(x_0, pos, class_labels, steps=10, guidance_scale=3.0)

        if self.train_on_latents:
            samples = 1 / 0.18215 * samples
            samples = self.img_vae.decode(samples.float()).sample

        image_grid = torchvision.utils.make_grid(samples, nrow=int(math.sqrt(num_images)), normalize=True, scale_each=True)
        fname = os.path.join(self.run_dir, f"sample-{self.cur_epoch}.png")
        torchvision.utils.save_image(image_grid, fname)

    def train(self, eval_interval: int):
        """Main training loop.

        :param log_interval: When to log the metrics.
        :param eval_interval: When to evaluate the model.
        :param save_interval: When to save the model.
        """

        self.accelerator.init_trackers(project_name="EGC")

        for epoch in range(self.num_epochs):
            self.cur_epoch = epoch
            self._train_one_epoch()

            if eval_interval > 0 and self.cur_epoch % eval_interval == 0:
                self.evaluate(self.net, self.test_dataloader)

        self.cur_epoch = self.num_epochs
        self.evaluate(self.net, self.test_dataloader)
        self._save("model-final")
        self._sample_images()

    def _gather(self, metrics: OrderedDict):
        """Gather the metrics across all processes."""
        global_metrics = self.accelerator.gather_for_metrics(metrics)
        if self.accelerator.is_local_main_process:
            global_metrics = {k: v.mean().item() for k, v in global_metrics.items()}
        return global_metrics

    @accelerator.on_local_main_process
    def _update_ema(self):
        """Update the EMA model."""
        self.ema.update()

    @torch.no_grad()
    def _encode_images_to_latents(self, images):
        images = self.img_vae.encode(images)["latent_dist"].sample()
        images = self.latent_scale_factor * images
        return images

    def _train_one_epoch(self):
        cls_loss_meter = Meter()
        cls_acc_meter = Meter()
        cls_ece_meter = Meter()
        mse_loss_meter = Meter()

        self.net.train()

        with tqdm(total=len(self.train_dataloader), desc=f"Epoch {self.cur_epoch}", disable=not self.accelerator.is_local_main_process, dynamic_ncols=True) as pbar:
            while True:
                patch_size = int(np.random.choice(self.patch_list, p=self.p_list))
                batch_mul = self.batch_mul_dict[patch_size] // self.batch_mul_dict[self.img_resolution]

                # NOTE: Check if batch_mul goes over the total number of batches, and adjust it
                if pbar.n + batch_mul > pbar.total:
                    batch_mul = pbar.total - pbar.n

                pbar.update(batch_mul)

                # ---------------------------------------------------------------------
                # Classification loss

                cls_images, cls_labels = next(self.cls_dataloader)

                if self.train_on_latents:
                    cls_images = self._encode_images_to_latents(cls_images)

                cls_images, cls_labels = get_patches(cls_images, self.img_resolution), torch.cat([cls_labels, cls_labels]).argmax(dim=1)

                with self.accelerator.no_sync(self.net):
                    logits, ce_loss, weighted_ce_loss = self.diffusion(cls_images, cls_labels, cls_mode=True)
                    acc = (logits.argmax(dim=1) == cls_labels).float().mean()
                    ece = self.ece(logits, cls_labels)

                    cls_loss_meter.update(ce_loss, cls_images.size(0))
                    cls_acc_meter.update(acc, cls_images.size(0))
                    cls_ece_meter.update(ece, cls_images.size(0))

                    self.accelerator.backward(self.ce_weight * weighted_ce_loss)

                # ---------------------------------------------------------------------
                # Diffusion loss

                images, labels = get_batch_data(self.train_dataloader, self.device, batch_mul)

                if self.train_on_latents:
                    images = self._encode_images_to_latents(images)

                images, labels = get_patches(images, patch_size), labels.argmax(dim=1)

                mse_loss = self.diffusion(images, labels)
                mse_loss_meter.update(mse_loss, images.size(0))

                self.accelerator.backward(mse_loss / batch_mul)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self._update_ema()

                if pbar.n >= pbar.total:
                    break

            grad_norm, param_norm = self._compute_norms()
            metrics = {
                "mse_loss": mse_loss_meter.compute().clone().detach(),
                "cls_loss": cls_loss_meter.compute().clone().detach(),
                "cls_acc": cls_acc_meter.compute().clone().detach(),
                "cls_ece": cls_ece_meter.compute().clone().detach(),
                "grad_norm": grad_norm,
                "param_norm": param_norm,
                "lr": torch.tensor(self.optimizer.param_groups[0]["lr"], device=self.device),
            }
            metrics = self._gather(metrics)
            pbar.set_postfix(metrics)
            pbar.close()

        self.accelerator.log(metrics, step=self.cur_epoch)

    @torch.no_grad()
    def evaluate(self, net: torch.nn.Module, dataloader: DataLoader):
        """Evaluate the model on the given dataset."""

        dataloader = self.accelerator.prepare(dataloader)
        loss_meter = Meter()
        acc_meter = Meter()
        ece_meter = Meter()

        net.eval()

        with tqdm(total=len(dataloader), desc="Validation", disable=not self.accelerator.is_local_main_process, dynamic_ncols=True) as pbar:
            for x, y in dataloader:
                pbar.update(1)

                y = y.argmax(dim=1)

                logits = net(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                acc = (logits.argmax(dim=1) == y).float().mean()
                ece = self.ece(logits, y)

                loss_meter.update(loss.mean(), x.size(0))
                acc_meter.update(acc, x.size(0))
                ece_meter.update(ece, x.size(0))

            metrics = {
                "val_cls_loss": loss_meter.compute().clone().detach(),
                "val_cls_acc": acc_meter.compute().clone().detach(),
                "val_cls_ece": ece_meter.compute().clone().detach(),
            }
            metrics = self._gather(metrics)
            pbar.set_postfix(metrics)
            pbar.close()

        net.train()

        self.accelerator.log(metrics, step=self.cur_epoch)

    @torch.no_grad()
    def get_probs(self, net: torch.nn.Module, dataloader: DataLoader):
        """Get the probability outputs of the model on the given dataset."""

        dataloader = self.accelerator.prepare(dataloader)
        probs = []

        net.eval()

        with tqdm(total=len(dataloader), desc="Getting Probabilities", disable=not self.accelerator.is_local_main_process, dynamic_ncols=True) as pbar:
            for x, _ in dataloader:
                pbar.update(1)
                logits = net(x)
                probs.append(logits.softmax(dim=1))
            pbar.close()

        net.train()

        probs = torch.cat(probs, dim=0)
        probs = torch.max(probs, dim=1).values

        return probs

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

    @accelerator.on_local_main_process
    def _save(self, filename: str):
        """Save the model and optimizer state."""

        data = {
            "epoch": self.cur_epoch,
            "net": self.accelerator.unwrap_model(self.net).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
        }

        torch.save(data, os.path.join(self.run_dir, f"{filename}.pt"))

    def _load(self):
        """Load the latest checkpoint from the directory or the specified file."""
        ckpt = self.resume_from
        if not ckpt.endswith(".pt"):
            pt_files = sorted(f for f in os.listdir(ckpt) if f.endswith(".pt"))
            ckpt = os.path.join(ckpt, pt_files[-1])

        self.print_fn(f"Resuming from {ckpt}...")

        data = torch.load(ckpt, weights_only=True)
        self.cur_epoch = data["epoch"]
        self.accelerator.unwrap_model(self.net).load_state_dict(data["net"])
        self.optimizer.load_state_dict(data["optimizer"])

        if self.accelerator.is_local_main_process:
            self.ema.load_state_dict(data["ema"])
            self.ema = self.ema.to(self.device)
