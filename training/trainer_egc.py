import copy
import json
import math
import os

import numpy as np
import torch
import torchvision
from accelerate import Accelerator
from diffusers import AutoencoderKL
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from tqdm import tqdm

import dnnlib

from .patch import get_batch_data, get_patches
from .trainer import BaseTrainer
from .utils import Meter

accelerator = Accelerator()


class EGCTrainer(BaseTrainer):
    """
    Trainer for EGC.

    A cleaned-up version of the main method proposed in
    [EGC: Image Generation and Classification via a Diffusion Energy-Based Model](https://openaccess.thecvf.com/content/ICCV2023/papers/Guo_EGC_Image_Generation_and_Classification_via_a_Diffusion_Energy-Based_Model_ICCV_2023_paper.pdf).

    This implementation also incorporates Patch Diffusion as proposed in
    [Patch Diffusion: Faster and More Data-Efficient Training of Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/e4667dd0a5a54b74019b72b677ed8ec1-Paper-Conference.pdf).
    """

    def __init__(self, diffusion_kwargs, num_steps: int = 100000, ce_weight: float = 0.001, real_p: float = 0.5, target: str = "epsilon", **trainer_kwargs):
        """
        Args:
            diffusion_kwargs (`dict`): Diffusion model options.
            num_steps (`int`): Total training steps.
            ce_weight (`float`): Classification loss weight.
            real_p (`float`): Probability of using full vs patch images.
            target (`str`): Diffusion target type.
            trainer_kwargs: Additional trainer options.
        """
        super().__init__(**trainer_kwargs)

        # EGC-specific attributes
        self.diffusion_kwargs = diffusion_kwargs
        self.num_steps = num_steps
        self.ce_weight = ce_weight
        self.real_p = real_p
        self.target = target

        self._init_trainer()

    def _init_trainer(self):
        self._init_env()
        self._build_network_and_diffusion()
        self._prepare_patch_info()

    def _build_network_and_diffusion(self):
        """Setup network and diffusion."""
        self.print_fn("Setting up network and diffusion...")

        if self.train_on_latents:
            self.img_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device)
            self.img_vae.eval()
            self._set_requires_grad(self.img_vae, False)
            self.img_resolution, self.img_channels = self.datamodule.img_resolution // 8, 4
        else:
            self.img_resolution, self.img_channels = self.datamodule.img_resolution, self.datamodule.img_channels

        attention_ds = []
        for res in self.network_kwargs["attn_resolutions"]:
            attention_ds.append(self.img_resolution // int(res))

        self.network_kwargs.update(
            {
                "attn_resolutions": tuple(attention_ds),
                "img_resolution": self.img_resolution,
                "in_channels": self.img_channels + 2,
                "out_channels": self.datamodule.label_dim,
                "label_dim": self.datamodule.label_dim,
            }
        )

        self.net = dnnlib.util.construct_class_by_name(**self.network_kwargs)
        self.diffusion = dnnlib.util.construct_class_by_name(**self.diffusion_kwargs, model=self.net)

        sampler_kwargs = copy.deepcopy(self.diffusion_kwargs)
        sampler_kwargs.update({"class_name": "training.diffusion.DDIMSampler"})

        if self.accelerator.is_main_process:
            with open(os.path.join(self.run_dir, "network_kwargs.json"), "w") as f:
                json.dump(self.network_kwargs, f)

            with open(os.path.join(self.run_dir, "sampler_kwargs.json"), "w") as f:
                json.dump(sampler_kwargs, f)

        # Setup EMA
        self.print_fn("Setting up EMA...")
        self.ema = EMA(self.net, power=3 / 4, update_after_step=1, update_every=1, include_online_model=False)
        self.sampler = dnnlib.util.construct_class_by_name(**sampler_kwargs, model=self.ema)

        # Setup the optimizer
        self.print_fn("Setting up optimizer...")
        self.optimizer = dnnlib.util.construct_class_by_name(params=self.net.parameters(), **self.optimizer_kwargs)

        # Prepare for distributed training
        self.net, self.ema, self.diffusion, self.optimizer = self.accelerator.prepare(self.net, self.ema, self.diffusion, self.optimizer)

    def _prepare_patch_info(self):
        """Prepare patch information for Patch Diffusion."""
        real_p = self.real_p
        img_resolution = self.img_resolution
        is_32 = img_resolution == 32

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

    @accelerator.on_main_process
    def _sample_images(self, filename: str, num_images=64):
        """Sample images from the EMA model and save them.

        Args:
            filename (`str`): Filename to save the images.
            num_images (`int`, optional): Number of images to sample. Defaults to 64.
        """
        self.sampler.model.eval()

        img_resolution = self.img_resolution
        label_dim = self.datamodule.label_dim
        img_channels = self.img_channels

        x_start, y_start, resolution, image_size = (0, 0, img_resolution, img_resolution)
        x_pos = torch.arange(x_start, x_start + image_size).view(1, -1).repeat(image_size, 1)
        y_pos = torch.arange(y_start, y_start + image_size).view(-1, 1).repeat(1, image_size)
        x_pos = (x_pos / (resolution - 1) - 0.5) * 2.0
        y_pos = (y_pos / (resolution - 1) - 0.5) * 2.0

        pos = torch.stack([x_pos, y_pos], dim=0).to(self.device)
        pos = pos.unsqueeze(0).repeat(num_images, 1, 1, 1)
        shape = (num_images, img_channels, img_resolution, img_resolution)
        x_0 = torch.randn(shape, device=self.device)
        class_labels = torch.randint(0, label_dim, (num_images,), device=self.device)

        samples = self.sampler(x_0, pos, class_labels, steps=10)

        if self.train_on_latents:
            latents = 1 / self.latent_scale_factor * samples
            samples = self.img_vae.decode(latents.float()).sample

        image_grid = torchvision.utils.make_grid(samples, nrow=int(math.sqrt(num_images)), normalize=True, scale_each=True)
        fname = os.path.join(self.run_dir, f"{filename}.png")
        torchvision.utils.save_image(image_grid, fname)

    def fit(self, log_interval: int, eval_interval: int, save_interval: int):
        """Main training loop.

        Args:
            log_interval (`int`): When to log the metrics.
            eval_interval (`int`): When to evaluate the model.
            save_interval (`int`): When to save the model.
        """
        self.best_val_ece = float("inf")
        self.best_val_loss = float("inf")
        self.cur_step = 0
        self.print_fn(f"Training for {self.num_steps} steps...")

        test_dataloader = self.datamodule.test_dataloader

        for step in range(self.num_steps):
            self.cur_step = step
            self._training_step(log_interval)

            if eval_interval > 0 and self.cur_step % eval_interval == 0:
                self.evaluate(self.ema, test_dataloader)

            if save_interval > 0 and self.cur_step % save_interval == 0:
                self._save_checkpoint(f"model-{self.cur_step}")
                self._sample_images(f"sample-{self.cur_step}")

        self.cur_step = self.num_steps
        self.evaluate(self.ema, test_dataloader)

        if not self.active_learning:
            self._save_checkpoint("model-final")
            self._sample_images("sample-final")

    @accelerator.on_main_process
    def _print_metrics(self, metrics: dict):
        """Print the metrics."""
        self.print_fn(f"\nStep {self.cur_step}/{self.num_steps}")
        for name, value in metrics.items():
            self.print_fn(f"{name} = {value:.6f}")

    def _training_step(self, log_interval: int):
        """Perform a single training step."""
        metrics = {}

        self.net.train()

        accum_cls_loss = torch.tensor(0.0, device=self.device)
        accum_cls_acc = torch.tensor(0.0, device=self.device)
        accum_cls_ece = torch.tensor(0.0, device=self.device)
        accum_mse_loss = torch.tensor(0.0, device=self.device)

        self.optimizer.zero_grad(set_to_none=True)

        for _ in range(self.accum_steps):
            cls_images, cls_labels = get_batch_data(self.datamodule.cls_dataloader)

            if self.train_on_latents:
                cls_images = self._encode_latents(cls_images)

            cls_images, cls_labels = get_patches(cls_images, self.img_resolution), torch.cat([cls_labels, cls_labels]).argmax(dim=1)

            with self.accelerator.no_sync(self.net):
                logits, ce_loss, weighted_ce_loss = self.diffusion(cls_images, cls_labels, cls_mode=True)
                acc = (logits.argmax(dim=1) == cls_labels).float().mean()
                ece = self.ece(logits, cls_labels).mean()

                weighted_ce_loss = weighted_ce_loss
                accum_cls_loss += ce_loss.mean()
                accum_cls_acc += acc
                accum_cls_ece += ece

                self.accelerator.backward(self.ce_weight * weighted_ce_loss)

            patch_size = int(np.random.choice(self.patch_list, p=self.p_list))
            batch_mul = self.batch_mul_dict[patch_size] // self.batch_mul_dict[self.img_resolution]

            images, labels = get_batch_data(self.datamodule.train_dataloader, batch_mul)

            if self.train_on_latents:
                images = self._encode_latents(images)

            images, labels = get_patches(images, patch_size), labels.argmax(dim=1)

            mse_loss = self.diffusion(images, labels)
            accum_mse_loss += mse_loss

            self.accelerator.backward(mse_loss / batch_mul)

        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)

        grad_norm, param_norm = self._compute_norms()

        self.accelerator.wait_for_everyone()
        self.optimizer.step()

        self._update_ema()

        metrics = {
            "cls_loss": (accum_cls_loss / self.accum_steps).clone().detach(),
            "cls_acc": (accum_cls_acc / self.accum_steps).clone().detach(),
            "cls_ece": (accum_cls_ece / self.accum_steps).clone().detach(),
            "mse_loss": (accum_mse_loss / self.accum_steps).clone().detach(),
            "grad_norm": grad_norm.clone().detach(),
            "param_norm": param_norm.clone().detach(),
            "lr": torch.tensor(self.optimizer.param_groups[0]["lr"], device=self.device),
        }
        metrics = {k: v.item() for k, v in metrics.items()}

        if self.cur_step % log_interval == 0:
            self._print_metrics(metrics)
            self.accelerator.log(metrics, step=self.cur_step + (self.al_mul * self.num_steps))

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
                labels = labels.argmax(dim=1)

                if self.train_on_latents:
                    images = self._encode_latents(images)

                images = get_patches(images, self.img_resolution)
                clean_timesteps = torch.zeros(images.shape[0], dtype=torch.long, device=self.device)

                logits = net(images, clean_timesteps, cls_mode=True)
                logits, labels = self.accelerator.gather((logits, labels))

                loss = torch.nn.functional.cross_entropy(logits, labels)
                acc = (logits.argmax(dim=1) == labels).float().mean()
                ece = self.ece(logits, labels)

                loss_meter.update(loss.mean(), images.size(0))
                acc_meter.update(acc, images.size(0))
                ece_meter.update(ece, images.size(0))

                pbar.update(1)

            metrics = {
                "val_cls_loss": loss_meter.compute().clone().detach(),
                "val_cls_acc": acc_meter.compute().clone().detach(),
                "val_cls_ece": ece_meter.compute().clone().detach(),
            }
            metrics = {k: v.item() for k, v in metrics.items()}
            pbar.set_postfix(metrics)
            pbar.close()

        if metrics["val_cls_ece"] < self.best_val_ece:
            self.best_val_ece = metrics["val_cls_ece"]
            filename = "model-best_val_ece" if not self.active_learning else f"al_iter_{self.al_mul+1}-best_val_ece"
            self.best_val_ece_path = os.path.join(self.run_dir, f"{filename}.pt")

            if self.accelerator.is_main_process:
                self._save_checkpoint(filename, metrics)

        if metrics["val_cls_loss"] < self.best_val_loss:
            self.best_val_loss = metrics["val_cls_loss"]
            filename = "model-best_val_loss" if not self.active_learning else f"al_iter_{self.al_mul+1}-best_val_loss"
            self.best_val_loss_path = os.path.join(self.run_dir, f"{filename}.pt")

            if self.accelerator.is_main_process:
                self._save_checkpoint(filename, metrics)

        self.accelerator.log(metrics, step=self.cur_step + (self.al_mul * self.num_steps))

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
                if self.train_on_latents:
                    images = self._encode_latents(images)

                images = get_patches(images, self.img_resolution)
                clean_timesteps = torch.zeros(images.shape[0], dtype=torch.long, device=self.device)

                logits = net(images, clean_timesteps, cls_mode=True)
                logits = self.accelerator.gather(logits)
                prob = logits.softmax(dim=1)
                probs.append(prob)

                pbar.update(1)

            pbar.close()

        net.train()

        return torch.cat(probs, dim=0)
