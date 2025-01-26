import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch_uncertainty.post_processing import TemperatureScaler

from training.trainer_egc import EGCTrainer
from training.trainer_wrn import WRNTrainer
from training.utils import cycle


class ActiveLearningTrainer:
    """Trainer for active learning."""

    def __init__(self, base_trainer, num_samples: float, calibrate=False, strategy="random"):
        """
        Args:
            base_trainer (`torch.nn.module`): Base trainer to use for active learning.
            num_samples (`float`): Number of samples to query at each iteration.
            calibrate (`bool`, optional, defaults to `False`): Whether to calibrate the model after each iteration.
            strategy (`str`, optional, defaults to `random`): Active learning strategy to use. Options are `random`, `lc`, `sm`, and `entropy`.
        """
        self.base_trainer = base_trainer
        self.num_samples = int(num_samples * len(self.base_trainer.cls_dataset))
        self.calibrate = calibrate
        self.strategy = strategy

        self.dataloader_kwargs = {"batch_size": 128, "num_workers": 4, "pin_memory": True, "drop_last": False}

        all_indices = np.array(list(range(len(self.base_trainer.cls_dataset))))
        self.labeled_indices = np.random.choice(all_indices, size=self.num_samples, replace=False)
        self.unlabeled_indices = np.setdiff1d(all_indices, self.labeled_indices)
        self._update_dataloaders()

    def random_query(self):
        """Query samples randomly.

        Returns:
            query_indices (np.ndarray): The indices of the samples to query.
        """
        query_size = min(self.num_samples, len(self.unlabeled_indices))
        query_indices = np.random.choice(self.unlabeled_indices, size=query_size, replace=False)

        return query_indices

    def least_confidence_query(self, net: torch.nn.Module):
        """Query samples using least confidence strategy.

        Args:
            net (torch.nn.Module): The model to use for querying.

        Returns:
            query_indices (np.ndarray): The indices of the samples to query.
        """
        unlabeled_dataset = Subset(self.base_trainer.cls_dataset, self.unlabeled_indices)
        dataloader = DataLoader(unlabeled_dataset, **self.dataloader_kwargs)

        probs = self.base_trainer.get_probs(net, dataloader)
        all_probs = self.base_trainer.accelerator.gather_for_metrics(probs)
        all_probs = all_probs[: len(self.unlabeled_indices)]
        lc_scores = 1 - torch.max(all_probs, dim=1).values

        query_size = min(self.num_samples, len(self.unlabeled_indices))
        sorted_indices = torch.argsort(lc_scores, descending=True)[:query_size].cpu().numpy()
        query_indices = self.unlabeled_indices[sorted_indices]

        return query_indices

    def _active_learning_step(self, net: torch.nn.Module):
        """Perform one active learning iteration.

        Args:
            net (torch.nn.Module): The model to use for querying.
        """

        match self.strategy:
            case "lc":
                query_indices = self.least_confidence_query(net)
            case "random":
                query_indices = self.random_query()
            case _:
                raise NotImplementedError(f"Active learning strategy {self.strategy} not implemented.")

        self.labeled_indices = np.concatenate([self.labeled_indices, query_indices])
        self.unlabeled_indices = np.setdiff1d(self.unlabeled_indices, query_indices)
        self._update_dataloaders()
        self.cur_al_iteration += 1

    def _log_distribution(self):
        """Log class distribution of labeled data."""

        labeled_dataset = Subset(self.base_trainer.cls_dataset, self.labeled_indices)
        dataloader = DataLoader(labeled_dataset, **self.dataloader_kwargs)
        distribution = torch.zeros(self.base_trainer.label_dim, dtype=torch.long)

        for _, targets in dataloader:
            targets = targets.argmax(dim=1)
            distribution += torch.bincount(targets.to(torch.long), minlength=self.base_trainer.label_dim)

        distribution = distribution.cpu().numpy().tolist()
        self.base_trainer.print_fn(f"Labeled: {len(self.labeled_indices)}, Unlabeled: {len(self.unlabeled_indices)}")
        self.base_trainer.print_fn(f"Class distribution: {distribution}")

        if self.base_trainer.accelerator.is_main_process:
            with open(f"{self.base_trainer.run_dir}/class_distribution.csv", "a") as f:
                f.write(",".join(map(str, distribution)) + "\n")

    def run_active_learning_loop(self, *args, **kwargs):
        """Run active learning loop."""

        self.cur_al_iteration = 1
        self._log_distribution()

        while True:
            self.base_trainer.print_fn(f"\nActive learning iteration: {self.cur_al_iteration}")
            self.base_trainer.train(*args, **kwargs)

            model = None
            if self.calibrate:
                self.base_trainer.print_fn("Calibrating model...")
                model = TemperatureScaler(model=self.base_trainer.net, device=self.base_trainer.device)
                model.fit(calibration_set=self.base_trainer.val_dataset)

            self.base_trainer._save_checkpoint(f"model-al_iter_{self.cur_al_iteration}-final")

            if hasattr(self.base_trainer, "_sample_images"):
                self.base_trainer._sample_images(f"al_iter-{self.cur_al_iteration}")

            self.base_trainer.accelerator.wait_for_everyone()

            if len(self.unlabeled_indices) == 0:
                self.base_trainer.print_fn("Unlabeled set is exhausted, stopping active learning...")
                break

            if model is None:
                model = self.base_trainer.net

            self._active_learning_step(model)
            self._log_distribution()
            self.base_trainer.al_mul += 1


class WRNActiveLearningTrainer(ActiveLearningTrainer):
    def __init__(self, num_samples: float, calibrate=False, strategy="random", **trainer_kwargs):
        base_trainer = WRNTrainer(**trainer_kwargs, active_learning=True)
        super().__init__(base_trainer, num_samples, calibrate, strategy)

    def _update_dataloaders(self):
        """Update dataloaders with the new labeled and unlabeled indices."""

        dataset = Subset(self.base_trainer.cls_dataset, self.labeled_indices)
        dataloader = DataLoader(dataset, **self.base_trainer.dataloader_kwargs)
        self.base_trainer.cls_dataloader = self.base_trainer.accelerator.prepare(dataloader)


class EGCActiveLearningTrainer(ActiveLearningTrainer):
    def __init__(self, num_samples: float, strategy="random", **trainer_kwargs):
        base_trainer = EGCTrainer(**trainer_kwargs, active_learning=True)
        super().__init__(base_trainer, num_samples, calibrate=False, strategy=strategy)

    def _update_dataloaders(self):
        """Update dataloaders with the new labeled and unlabeled indices."""

        dataset = Subset(self.base_trainer.cls_dataset, self.labeled_indices)
        dataloader = DataLoader(dataset, **self.base_trainer.dataloader_kwargs)
        self.base_trainer.cls_dataloader = self.base_trainer.accelerator.prepare(dataloader)

        # NOTE: EGC is trained in a semi-supervised manner
        self.base_trainer.cls_dataloader = cycle(self.base_trainer.cls_dataloader)
