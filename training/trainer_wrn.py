import json
import math
import os

import torch
from diffusers import AutoencoderKL
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import LBFGS
import dnnlib

from .trainer import BaseTrainer
from .utils import Meter
from .ece import ModelWithTemperature


class WRNTrainer(BaseTrainer):
    """Trainer for Wide Residual Network."""

    def __init__(self, num_epochs: int = 200, warmup_steps: int = 0, calibrate: bool = False, **trainer_kwargs):
        """
        Args:
            num_epochs (`int`): Total training epochs. Defaults to 200.
            warmup_steps (`int`): Number of warmup steps. Defaults to 0.
            calibrate (`bool`): Whether to calibrate model temperatures during evaluation. Defaults to `False`.
            trainer_kwargs (`dict`): Additional trainer options.
        """
        super().__init__(**trainer_kwargs)

        # WRN-specific attributes
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.calibrate = calibrate
        self.temperature = 1.0

        # NOTE: Experimental
        # self.best_score = 0.0

        self._init_trainer()

    def _init_trainer(self):
        self._init_env()
        self._build_network()

    def _build_network(self):
        """Setup network."""
        self.print_fn("Setting up network...")

        if self.train_on_latents:
            self.img_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device)
            self.img_vae.eval()
            self._set_requires_grad(self.img_vae, False)
            self.datamodule.img_resolution, self.datamodule.img_channels = self.datamodule.img_resolution // 8, 4

        self.network_kwargs.update({"label_dim": self.datamodule.label_dim, "in_channels": self.datamodule.img_channels})
        self.net = dnnlib.util.construct_class_by_name(**self.network_kwargs)

        if self.accelerator.is_main_process:
            with open(os.path.join(self.run_dir, "network_kwargs.json"), "w") as f:
                json.dump(self.network_kwargs, f, indent=4)

        # Setup the EMA
        self.ema = EMA(self.net, power=3 / 4, update_after_step=1, update_every=1, include_online_model=False)

        # Setup the optimizer
        self.print_fn("Setting up optimizer...")
        self.optimizer = dnnlib.util.construct_class_by_name(params=self.net.parameters(), **self.optimizer_kwargs)

        # Prepare for distributed training
        self.net, self.ema, self.optimizer = self.accelerator.prepare(self.net, self.ema, self.optimizer)

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
            decay_steps = (self.num_epochs * len(self.datamodule.cls_dataloader)) - self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.optimizer_kwargs.lr * (0.5 * (1 + math.cos(math.pi * decay_iter / decay_steps)))

    def fit(self, eval_interval: int):
        """Main training loop.

        Args:
            eval_interval (`int`): When to evaluate the model.
        """
        self.cur_iter = 0
        self.best_val_ece = float("inf")
        self.best_val_loss = float("inf")
        test_dataloader = self.datamodule.test_dataloader

        for epoch in range(self.num_epochs):
            self.cur_epoch = epoch
            self._train_one_epoch()

            if eval_interval > 0 and self.cur_epoch % eval_interval == 0:
                self.evaluate(self.ema, test_dataloader)

        if not self.active_learning:
            self._save_checkpoint("model-final")

    def _train_one_epoch(self):
        """Train the model for one epoch."""
        cls_loss_meter = Meter()
        cls_acc_meter = Meter()
        cls_ece_meter = Meter()

        self.net.train()

        dataloader = self.datamodule.cls_dataloader

        with tqdm(total=len(dataloader), desc=f"Epoch {self.cur_epoch}", disable=not self.accelerator.is_local_main_process, dynamic_ncols=True) as pbar:
            for images, labels in dataloader:
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

                self.optimizer.zero_grad(set_to_none=True)
                self.accelerator.backward(loss)
                self.optimizer.step()

                self._update_ema()
                self._step_lr()
                self.cur_iter += 1
                pbar.update(1)

            metrics = {
                "cls_loss": cls_loss_meter.compute().clone().detach(),
                "cls_acc": cls_acc_meter.compute().clone().detach(),
                "cls_ece": cls_ece_meter.compute().clone().detach(),
                "lr": torch.tensor(self.optimizer.param_groups[0]["lr"], device=self.device),
            }
            metrics = {k: v.item() for k, v in metrics.items()}
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
                labels = labels.argmax(dim=1)

                if self.train_on_latents:
                    images = self._encode_latents(images)

                logits = net(images)
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

        # NOTE: Experimental
        # score = metrics["val_cls_acc"] - (0.5 * metrics["val_cls_ece"])

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
                if self.train_on_latents:
                    images = self._encode_latents(images)

                logits = net(images) / self.temperature
                logits = self.accelerator.gather(logits)
                prob = logits.softmax(dim=1)
                probs.append(prob)

                pbar.update(1)

            pbar.close()

        net.train()

        return torch.cat(probs, dim=0)

    @torch.no_grad()
    def set_temperature(self, net: torch.nn.Module, dataloader: DataLoader):
        """Tune the temperature of the model (using the given dataloader).

        Reference: https://github.com/gpleiss/temperature_scaling (MIT License)

        Args:
            net (`torch.nn.Module`): The model to tune the temperature for.
            dataloader (`DataLoader`): The dataloader to use for tuning the temperature.
        """
        tempered_model = ModelWithTemperature(net)
        tempered_model = tempered_model.to(self.device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []

        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device).argmax(dim=1)

            if self.train_on_latents:
                images = self._encode_latents(images)

            logits = net(images)
            logits_list.append(logits)
            labels_list.append(labels)

        logits = torch.cat(logits_list).to(self.device)
        labels = torch.cat(labels_list).to(self.device)

        # Next: optimize the temperature w.r.t. NLL
        optimizer = LBFGS([tempered_model.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(tempered_model.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        self.print_fn(f"Optimal temperature: {tempered_model.temperature.item():.3f}")

        self.temperature = tempered_model.temperature.item()
