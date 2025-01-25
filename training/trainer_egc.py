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
from .utils import Meter, cycle

accelerator = Accelerator()


class EGCTrainer(BaseTrainer):
    """
    Trainer for EGC.

    A more cleaned-up version of the main method proposed in
    [EGC: Image Generation and Classification via a Diffusion Energy-Based Model](https://openaccess.thecvf.com/content/ICCV2023/papers/Guo_EGC_Image_Generation_and_Classification_via_a_Diffusion_Energy-Based_Model_ICCV_2023_paper.pdf).

    This implementation also incorporates Patch Diffusion as proposed in
    [Patch Diffusion: Faster and More Data-Efficient Training of Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/e4667dd0a5a54b74019b72b677ed8ec1-Paper-Conference.pdf).
    """

    def __init__(
        self,
        run_dir: str,
        dataset_kwargs,
        val_dataset_kwargs,
        test_dataset_kwargs,
        network_kwargs,
        diffusion_kwargs,
        optimizer_kwargs,
        num_steps: int = 100000,
        accum_steps: int = 1,
        batch_size: int = 128,
        ce_weight: float = 0.001,
        real_p: float = 0.5,
        target: str = "epsilon",
        train_on_latents: bool = False,
        seed: int = 1,
        resume_from: str = None,
        active_learning: bool = False,
    ):
        """
        Args:
            diffusion_kwargs (`dict`): Diffusion model options.
            num_steps (`int`): Total training steps.
            ce_weight (`float`): Classification loss weight.
            real_p (`float`): Probability of using full vs patch images.
            target (`str`): Diffusion target type.
            active_learning (`bool`): Whether to use active learning.
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

        # EGC-specific attributes
        self.diffusion_kwargs = diffusion_kwargs
        self.num_steps = num_steps
        self.ce_weight = ce_weight
        self.real_p = real_p
        self.target = target
        self.active_learning = active_learning

        self._init_trainer()

    def _init_trainer(self):
        self._init_env()
        self._prepare_datasets()
        self._prepare_dataloaders()
        self._build_network_and_diffusion()
        self._prepare_patch_info()
        self._load_checkpoint()

        # NOTE: We need to initialize the diffusion model after loading the checkpoint if it exists
        self.diffusion = dnnlib.util.construct_class_by_name(**self.diffusion_kwargs, model=self.accelerator.unwrap_model(self.net))
        self.diffusion = self.accelerator.prepare(self.diffusion)

    def _prepare_dataloaders(self):
        """Prepare datasets and dataloaders."""
        self.print_fn("Preparing dataloaders...")
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

        self.train_dataloader, self.cls_dataloader = cycle(self.train_dataloader), cycle(self.cls_dataloader)

    def _prepare_patch_info(self):
        """Prepare patch information for Patch Diffusion."""
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

    def _build_network_and_diffusion(self):
        """Setup network and diffusion."""
        self.print_fn("Setting up network and diffusion...")

        if self.train_on_latents:
            self.img_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device)
            self.img_vae.eval()
            self._set_requires_grad(self.img_vae, False)
            self.img_resolution, self.img_channels = self.img_resolution // 8, 4

        attention_ds = []
        for res in self.network_kwargs["attn_resolutions"]:
            attention_ds.append(self.img_resolution // int(res))

        self.network_kwargs.update(
            {
                "attn_resolutions": tuple(attention_ds),
                "img_resolution": self.img_resolution,
                "in_channels": self.img_channels + 2,
                "out_channels": self.label_dim,
                "label_dim": self.label_dim,
            }
        )
        self.net = dnnlib.util.construct_class_by_name(**self.network_kwargs)

        sampler_kwargs = copy.deepcopy(self.diffusion_kwargs)
        sampler_kwargs.update({"class_name": "training.diffusion.DDIMSampler"})

        if self.accelerator.is_main_process:
            with open(os.path.join(self.run_dir, "network_kwargs.json"), "w") as f:
                json.dump(self.network_kwargs, f)

            with open(os.path.join(self.run_dir, "sampler_kwargs.json"), "w") as f:
                json.dump(sampler_kwargs, f)

        # Setup EMA
        self.print_fn("Setting up EMA...")
        if self.accelerator.is_main_process:
            self.ema = EMA(self.net, power=3 / 4, include_online_model=False)
            self.sampler = dnnlib.util.construct_class_by_name(**sampler_kwargs, model=self.ema.ema_model)

        # Setup the optimizer
        self.print_fn("Setting up optimizer...")
        self.optimizer = dnnlib.util.construct_class_by_name(params=self.net.parameters(), **self.optimizer_kwargs)

        # Prepare for distributed training
        self.net, self.optimizer = self.accelerator.prepare(self.net, self.optimizer)

    @accelerator.on_main_process
    def _sample_images(self, filename: str, num_images=64):
        """Sample images from the EMA model and save them.

        Args:
            filename (`str`): Filename to save the images.
            num_images (`int`, optional): Number of images to sample. Defaults to 64.
        """
        self.ema.ema_model.eval()

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

        samples = self.sampler(x_0, pos, class_labels, steps=10)

        if self.train_on_latents:
            latents = 1 / self.latent_scale_factor * samples
            samples = self.img_vae.decode(latents.float()).sample

        image_grid = torchvision.utils.make_grid(samples, nrow=int(math.sqrt(num_images)), normalize=True, scale_each=True)
        fname = os.path.join(self.run_dir, f"{filename}.png")
        torchvision.utils.save_image(image_grid, fname)

    def train(self, log_interval: int, eval_interval: int, save_interval: int):
        """Main training loop.

        Args:
            log_interval (`int`): When to log the metrics.
            eval_interval (`int`): When to evaluate the model.
            save_interval (`int`): When to save the model.
        """

        self.best_val_loss = float("inf")
        self.cur_step = 0
        self.print_fn(f"Training for {self.num_steps - self.cur_step} steps...")

        for step in range(self.num_steps):
            self.cur_step = step
            self._training_step(log_interval)

            if eval_interval > 0 and self.cur_step % eval_interval == 0:
                self.evaluate(self.net, self.val_dataloader)

            if save_interval > 0 and self.cur_step % save_interval == 0:
                self._save_checkpoint(f"model-{self.cur_step}")
                self._sample_images(f"sample-{self.cur_step}")

        self.cur_step = self.num_steps
        self.evaluate(self.net, self.val_dataloader)

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
        self.optimizer.zero_grad(set_to_none=True)

        if self.ce_weight > 0:
            cls_images, cls_labels = get_batch_data(self.cls_dataloader)

            if self.train_on_latents:
                cls_images = self._encode_latents(cls_images)

            cls_images, cls_labels = get_patches(cls_images, self.img_resolution), torch.cat([cls_labels, cls_labels]).argmax(dim=1)

            with self.accelerator.no_sync(self.net):
                logits, ce_loss, weighted_ce_loss = self.diffusion(cls_images, cls_labels, cls_mode=True)
                acc = (logits.argmax(dim=1) == cls_labels).float().mean()
                ece = self.ece(logits, cls_labels)

                self.accelerator.backward(self.ce_weight * weighted_ce_loss)

        patch_size = int(np.random.choice(self.patch_list, p=self.p_list))
        batch_mul = self.batch_mul_dict[patch_size] // self.batch_mul_dict[self.img_resolution]

        images, labels = get_batch_data(self.train_dataloader, batch_mul)
        images, labels = images.to(self.device), labels.to(self.device)

        if self.train_on_latents:
            images = self._encode_latents(images)

        images, labels = get_patches(images, patch_size), labels.argmax(dim=1)

        mse_loss = self.diffusion(images, labels)

        self.accelerator.backward(mse_loss / batch_mul)

        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)

        self.accelerator.wait_for_everyone()
        self.optimizer.step()
        self._update_ema()

        grad_norm, param_norm = self._compute_norms()
        metrics = {
            "cls_loss": ce_loss.mean().clone().detach(),
            "cls_acc": acc.clone().detach(),
            "cls_ece": ece.clone().detach(),
            "mse_loss": mse_loss.mean().clone().detach(),
            "grad_norm": grad_norm.clone().detach(),
            "param_norm": param_norm.clone().detach(),
            "lr": torch.tensor(self.optimizer.param_groups[0]["lr"], device=self.device),
        }

        metrics = self._gather(metrics)
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
                pbar.update(1)

                labels = labels.argmax(dim=1)

                if self.train_on_latents:
                    images = self._encode_latents(images)

                images = get_patches(images, self.img_resolution)
                clean_timesteps = torch.zeros(images.shape[0], dtype=torch.long, device=self.device)

                logits = net(images, clean_timesteps, cls_mode=True)
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

        if self.accelerator.is_main_process and metrics["val_cls_loss"] < self.best_val_loss:
            self.print_fn(f"Saving best model with val loss: {metrics['val_cls_loss']:.4f}")
            self.best_val_loss = metrics["val_cls_loss"]

            filename = "model-best" if not self.active_learning else f"model-al_iter_{self.al_mul+1}-best"
            self._save_checkpoint(filename, {"val_loss": self.best_val_loss})

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
                pbar.update(1)

                if self.train_on_latents:
                    images = self._encode_latents(images)

                images = get_patches(images, self.img_resolution)
                clean_timesteps = torch.zeros(images.shape[0], dtype=torch.long, device=self.device)

                logits = net(images, clean_timesteps, cls_mode=True)
                prob = logits.softmax(dim=1)
                probs.append(prob)

            pbar.close()

        net.train()

        return torch.cat(probs, dim=0)
