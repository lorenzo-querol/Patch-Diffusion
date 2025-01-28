import json
import math
import os

import torch
from diffusers import AutoencoderKL
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from torch_uncertainty.post_processing import TemperatureScaler
from tqdm import tqdm

import dnnlib

from .trainer import BaseTrainer
from .utils import Meter


class WRNTrainer(BaseTrainer):
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
        warmup_steps: int = 0,
        num_epochs: int = 200,
        accum_steps: int = 1,
        batch_size: int = 128,
        seed: int = 1,
        resume_from: str = None,
        calibrate: bool = False,
        train_on_latents: bool = False,
        active_learning: bool = False,
    ):
        """
        Args:
            decay_epochs (`list[int]`): When to decay learning rate.
            decay_rate (`float`): Learning rate decay factor.
            num_epochs (`int`): Total training epochs.
            calibrate (`bool`): Whether to calibrate model temperatures.
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

        # WRN-specific attributes
        self.decay_epochs = decay_epochs
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
        self.num_epochs = num_epochs
        self.calibrate = calibrate
        self.active_learning = active_learning
        self.best_score = 0.0

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
        self.print_fn("Preparing dataloaders...")
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

    def _step_lr(self):
        """Step the learning rate. Uses cosine annealing with warmup.

        If `self.warmup_steps > 0`, then the learning rate is linearly increased from 0 to the initial learning rate.
        Else, the learning rate is decayed using cosine annealing.
        """

        if self.warmup_steps > 0 and self.cur_iter < self.warmup_steps:
            warmup_lr = self.optimizer_kwargs.lr * float(self.cur_iter) / float(self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = warmup_lr
        elif self.warmup_steps > 0 and self.cur_iter >= self.warmup_steps:
            decay_iter = self.cur_iter - self.warmup_steps
            decay_steps = (self.num_epochs * len(self.cls_dataloader)) - self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.optimizer_kwargs.lr * (0.5 * (1 + math.cos(math.pi * decay_iter / decay_steps)))

    def train(self, eval_interval: int):
        """Main training loop.

        Args:
            eval_interval (`int`): When to evaluate the model.
        """
        # self.best_val_ece = float("inf")
        # self.best_score = 0.0
        self.cur_iter = 0

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

        if not self.active_learning:
            self._save_checkpoint("model-final")

    def _train_one_epoch(self):
        """Train the model for one epoch."""
        cls_loss_meter = Meter()
        cls_acc_meter = Meter()
        cls_ece_meter = Meter()

        self.net.train()

        with tqdm(total=len(self.cls_dataloader), desc=f"Epoch {self.cur_epoch}", disable=not self.accelerator.is_local_main_process, dynamic_ncols=True) as pbar:
            for images, labels in self.cls_dataloader:
                pbar.update(1)

                if self.train_on_latents:
                    images = self._encode_latents(images)

                labels = labels.argmax(dim=1)
                logits = self.net(images)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                acc = (logits.argmax(dim=1) == labels).float().mean()
                ece = self.ece(logits, labels)

                cls_loss_meter.update(loss.mean(), images.size(0))
                cls_acc_meter.update(acc, images.size(0))
                cls_ece_meter.update(ece, images.size(0))

                self.optimizer.zero_grad(set_to_none=True)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self._update_ema()
                self._step_lr()
                self.cur_iter += 1

            metrics = {
                "cls_loss": cls_loss_meter.compute().clone().detach(),
                "cls_acc": cls_acc_meter.compute().clone().detach(),
                "cls_ece": cls_ece_meter.compute().clone().detach(),
                "lr": torch.tensor(self.optimizer.param_groups[0]["lr"], device=self.device),
            }
            metrics = self._gather(metrics)
            pbar.set_postfix(metrics)
            pbar.close()

        self.accelerator.log(metrics, step=self.cur_epoch + (self.al_mul * self.num_epochs))

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

        lambda_value = 0.5
        score = metrics["val_cls_acc"] - (lambda_value * metrics["val_cls_ece"])
        # if self.accelerator.is_main_process and metrics["val_cls_ece"] < self.best_val_ece:
        if self.accelerator.is_main_process and score > self.best_score:
            self.best_score = score
            self.print_fn(f"Saving best model ({self.best_score:.4f}) with val_acc: {metrics['val_cls_acc']:.4f}, val_ece: {metrics['val_cls_ece']:.4f}")

            filename = f"model-best-{self.cur_epoch}" if not self.active_learning else f"model-al_iter_{self.al_mul+1}-best-{self.cur_epoch}"
            self._save_checkpoint(filename, {"val_cls_ece": metrics["val_cls_ece"], "val_cls_acc": metrics["val_cls_acc"]})

        metrics.update({"custom_score": score})
        self.accelerator.log(metrics, step=self.cur_epoch + (self.al_mul * self.num_epochs))

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
