# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from calibration.ece import ECELoss
from torch_utils import persistence
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Patch version of the loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


@persistence.persistent_class
class PatchEDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.ece_criterion = ECELoss()

    def patchify(self, images, patch_size, padding=None):
        device = images.device
        batch_size, resolution = images.size(0), images.size(2)

        if padding is not None:
            padded = torch.zeros((images.size(0), images.size(1), images.size(2) + padding * 2, images.size(3) + padding * 2), dtype=images.dtype, device=device)
            padded[:, :, padding:-padding, padding:-padding] = images
        else:
            padded = images

        h, w = padded.size(2), padded.size(3)
        th, tw = patch_size, patch_size
        if w == tw and h == th:
            i = torch.zeros((batch_size,), device=device).long()
            j = torch.zeros((batch_size,), device=device).long()
        else:
            i = torch.randint(0, h - th + 1, (batch_size,), device=device)
            j = torch.randint(0, w - tw + 1, (batch_size,), device=device)

        rows = torch.arange(th, dtype=torch.long, device=device) + i[:, None]
        columns = torch.arange(tw, dtype=torch.long, device=device) + j[:, None]
        padded = padded.permute(1, 0, 2, 3)
        padded = padded[:, torch.arange(batch_size)[:, None, None], rows[:, torch.arange(th)[:, None]], columns[:, None]]
        padded = padded.permute(1, 0, 2, 3)

        x_pos = torch.arange(tw, dtype=torch.long, device=device).unsqueeze(0).repeat(th, 1).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        y_pos = torch.arange(th, dtype=torch.long, device=device).unsqueeze(1).repeat(1, tw).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        x_pos = x_pos + j.view(-1, 1, 1, 1)
        y_pos = y_pos + i.view(-1, 1, 1, 1)
        x_pos = (x_pos / (resolution - 1) - 0.5) * 2.0
        y_pos = (y_pos / (resolution - 1) - 0.5) * 2.0
        images_pos = torch.cat((x_pos, y_pos), dim=1)

        return padded, images_pos

    def __call__(self, net, images, patch_size, labels=None, augment_pipe=None, cls_mode=False, eval_mode=False):
        images, images_pos = self.patchify(images, patch_size)

        if eval_mode:
            clean_sigma = torch.zeros(images.shape[0], device=images.device)
            logits = net(images, clean_sigma, x_pos=images_pos, class_labels=labels, cls_mode=cls_mode, eval_mode=eval_mode)
            ce_loss = F.cross_entropy(logits, labels.argmax(dim=1), reduction="none")
            return logits, ce_loss

        # Noise distribution.
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        # Loss weighting.
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

        # Add noise to the input.
        n = torch.randn_like(y) * sigma
        yn = y + n

        if cls_mode:
            logits = net(yn, sigma, x_pos=images_pos, class_labels=labels, cls_mode=cls_mode, eval_mode=eval_mode)
            cls_weight = weight / weight.max()

            ce_loss = F.cross_entropy(logits, labels.argmax(dim=1), reduction="none")
            ce_loss = (ce_loss * cls_weight).mean()

            correct = (logits.argmax(dim=1) == labels.argmax(dim=1)).float().mean()
            ece = self.ece_criterion(logits, labels.argmax(dim=1))

            return ce_loss, correct, ece

        # Predicted score is the output of the network.
        D_yn = net(yn, sigma, x_pos=images_pos, class_labels=labels, augment_labels=augment_labels)

        # Compute loss.
        loss = weight * ((D_yn - y) ** 2)

        return loss


# ----------------------------------------------------------------------------
