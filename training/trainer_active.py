import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch_uncertainty.post_processing import TemperatureScaler

from training.trainer_egc import Trainer as EGCTrainer
from training.trainer_wrn import Trainer as WRNTrainer


class ActiveLearningTrainer:
    def __init__(self, base_trainer, num_samples: float, calibrate=False, strategy="lc", **trainer_kwargs):
        self.base_trainer = base_trainer
        self.calibrate = calibrate
        self.temp_scale_ratio = 0.1

        self.num_samples = int(num_samples * len(self.base_trainer.cls_dataset))
        self.strategy = strategy

        all_indices = np.array(list(range(len(self.base_trainer.cls_dataset))))

        self.labeled_indices = np.random.choice(all_indices, size=self.num_samples, replace=False)
        self.unlabeled_indices = np.setdiff1d(all_indices, self.labeled_indices)
        self._update_dataloaders()

    def _update_dataloaders(self):
        loader = DataLoader(
            self.base_trainer.cls_dataset,
            sampler=SubsetRandomSampler(self.labeled_indices),
            **self.base_trainer.dataloader_kwargs,
        )
        self.base_trainer.cls_dataloader = self.base_trainer.accelerator.prepare(loader)

    def _create_calibration_set(self):
        calibration_size = int(len(self.labeled_indices) * self.temp_scale_ratio)
        self.base_trainer.print_fn(f"Creating calibration loader with {calibration_size} samples...")

        train_indices, cal_indices = random_split(self.labeled_indices, [len(self.labeled_indices) - calibration_size, calibration_size])
        self.labeled_indices = train_indices.indices
        self._update_dataloaders()

        cal_dataset = torch.utils.data.Subset(self.base_trainer.cls_dataset, cal_indices.indices)

        return cal_dataset

    def query_samples(self, net):
        """Query samples using least confidence"""
        dataloader = DataLoader(
            self.base_trainer.cls_dataset,
            sampler=SubsetRandomSampler(self.unlabeled_indices),
            batch_size=self.base_trainer.batch_size,
            pin_memory=True,
        )

        probs = self.base_trainer.get_probs(net, dataloader)
        probs = self.base_trainer.accelerator.gather_for_metrics(probs)

        num_to_sample = min(self.num_samples, len(self.unlabeled_indices))

        if num_to_sample == 0:
            return []

        # Ensure sorted_indices is within bounds
        sorted_indices = torch.argsort(probs)[: len(self.unlabeled_indices)].cpu().numpy().tolist()
        sorted_indices = sorted_indices[:num_to_sample]  # Limit to num_to_sample
        query_indices = [self.unlabeled_indices[i] for i in sorted_indices]

        return query_indices

    def random_query_samples(self):
        """Query samples randomly"""
        num_to_sample = min(self.num_samples, len(self.unlabeled_indices))
        query_indices = np.random.choice(self.unlabeled_indices, size=num_to_sample, replace=False)

        return query_indices

    def _active_learning_step(self, net):
        """Perform one active learning iteration."""
        if self.strategy == "lc":
            query_indices = self.query_samples(net)
        elif self.strategy == "random":
            query_indices = self.random_query_samples()

        self.labeled_indices.extend(query_indices)
        self.unlabeled_indices = [i for i in self.unlabeled_indices if i not in query_indices]
        self._update_dataloaders()
        self.cur_al_iteration += 1

    def _log_distribution(self):
        """Log class distribution of labeled data."""
        loader = DataLoader(
            self.base_trainer.cls_dataset,
            sampler=SubsetRandomSampler(self.labeled_indices),
            batch_size=self.base_trainer.batch_size,
            shuffle=False,
        )

        distribution = torch.zeros(self.base_trainer.label_dim, dtype=torch.long)
        for _, targets in loader:
            targets = targets.argmax(dim=1)
            distribution += torch.bincount(targets.to(torch.long), minlength=self.base_trainer.label_dim)

        self.base_trainer.print_fn(f"Labeled: {len(self.labeled_indices)}, Unlabeled: {len(self.unlabeled_indices)}")
        self.base_trainer.print_fn(f"Class distribution: {distribution.cpu().numpy().tolist()}")

    def _save_distribution(self, indices, tag):
        """Save class distribution."""
        loader = DataLoader(
            self.base_trainer.cls_dataset,
            sampler=SubsetRandomSampler(indices),
            batch_size=self.base_trainer.batch_size,
            shuffle=False,
        )

        distribution = torch.zeros(self.base_trainer.label_dim, dtype=torch.long)
        for _, targets in loader:
            targets = targets.argmax(dim=1)
            distribution += torch.bincount(targets.to(torch.long), minlength=self.base_trainer.label_dim)

        if not self.base_trainer.accelerator.is_main_process:
            return

        csv_path = os.path.join(self.base_trainer.run_dir, f"dist_{tag}.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode="a+", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["iter", *list(range(self.base_trainer.label_dim))])

            writer.writerow([self.cur_al_iteration, *distribution.cpu().numpy().tolist()])

    def _log_per_class_accuracy(self, per_class_acc, tag):
        """Log per class accuracy."""
        gathered_values = self.base_trainer.accelerator.gather_for_metrics(per_class_acc).chunk(self.base_trainer.accelerator.num_processes, dim=0)
        averaged_values = torch.mean(torch.stack(gathered_values), dim=0).cpu().numpy().tolist()

        self.base_trainer.print_fn(f"({tag}) Per class accuracy:")
        for class_idx, acc in enumerate(averaged_values):
            self.base_trainer.print_fn(f"Class {class_idx}: {acc:.4f}")

    def _save_accuracies(self, per_class_acc, tag):
        if not self.base_trainer.accelerator.is_main_process:
            return

        gathered_values = self.base_trainer.accelerator.gather_for_metrics(per_class_acc).chunk(self.base_trainer.accelerator.num_processes, dim=0)
        averaged_values = torch.mean(torch.stack(gathered_values), dim=0).cpu().numpy().tolist()
        averaged_values = [f"{acc:.4f}" for acc in averaged_values]

        csv_path = os.path.join(self.base_trainer.run_dir, f"per_class_accuracy_{tag}.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode="a+", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["iter", *range(len(averaged_values))])

            writer.writerow([self.cur_al_iteration, *averaged_values])

    def _save_metrics(self, metrics, tag):
        if not self.base_trainer.accelerator.is_main_process:
            return

        global_metrics = {}
        for name, value in metrics.items():
            gathered_values = self.base_trainer.accelerator.gather_for_metrics(value)
            global_avg = gathered_values.mean().item()
            global_metrics[name] = f"{global_avg:.4f}"

        csv_path = os.path.join(self.base_trainer.run_dir, f"metrics_{tag}.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode="a+", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["iter", *global_metrics.keys()])

            writer.writerow([self.cur_al_iteration, *global_metrics.values()])

    def _reset_environment(self):
        self.base_trainer.cur_epoch = 0

        for param_group in self.base_trainer.optimizer.param_groups:
            param_group["lr"] = self.base_trainer.optimizer_kwargs["lr"]

    def run_active_learning(self, *args, **kwargs):
        """Run active learning loop"""

        self.cur_al_iteration = 1

        while True:
            self.base_trainer.print_fn(f"\nActive learning iteration: {self.cur_al_iteration}")

            # if self.calibrate:
            #     calibration_set = self._create_calibration_set()

            self._log_distribution()
            # self._save_distribution(self.unlabeled_indices, "unlabeled")
            # self._save_distribution(self.labeled_indices, "labeled")

            self.base_trainer.train(*args, **kwargs)

            if self.calibrate:
                self.base_trainer.print_fn("Calibrating model...")
                calibrated_model = TemperatureScaler(model=self.base_trainer.net, device=self.base_trainer.device)
                calibrated_model.fit(calibration_set=self.base_trainer.val_dataset)

            # metrics, per_class_acc, _ = self.base_trainer.evaluate(calibrated_model, self.base_trainer.val_dataloader, return_confidences=True, return_per_class_acc=True)

            # self._save_metrics(metrics, "val")
            # self._log_per_class_accuracy(per_class_acc, "val")
            # self._save_accuracies(per_class_acc, "val")

            # self.base_trainer._report_metrics(metrics)
            self.base_trainer._save(f"al_iter-{self.cur_al_iteration}")

            self._reset_environment()

            if len(self.unlabeled_indices) == 0:
                self.base_trainer.print_fn("Unlabeled set is exhausted, active learning completed.")
                self.base_trainer.accelerator.end_training()
                break

            self._active_learning_step(calibrated_model if self.calibrate else self.base_trainer.net)


class WRNActiveLearningTrainer(ActiveLearningTrainer):
    def __init__(self, num_samples: float, calibrate=False, strategy="lc", **trainer_kwargs):
        base_trainer = WRNTrainer(**trainer_kwargs)
        super().__init__(base_trainer, num_samples, calibrate, strategy, **trainer_kwargs)


class EGCActiveLearningTrainer(ActiveLearningTrainer):
    def __init__(self, num_samples: float, **trainer_kwargs):
        base_trainer = EGCTrainer(**trainer_kwargs)
        super().__init__(base_trainer, num_samples, **trainer_kwargs)
