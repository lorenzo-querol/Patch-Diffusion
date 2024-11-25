import numpy as np
import torch

from torch import distributed as dist
from calibration.ece import ECE
from training.patch import get_patches
from tqdm import tqdm


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def evaluate(net, dataloader, device):
    losses, accs, eces = [], [], []
    ece_criterion = ECE()
    criterion = torch.nn.CrossEntropyLoss()

    net.eval()

    with torch.no_grad():
        for images, labels in tqdm(dataloader, disable=not dist.get_rank() == 0):
            images, labels = images.to(device), labels.to(device)
            patches, x_pos = get_patches(images, images.shape[-1])
            t_cls = np.random.choice(1, size=(images.shape[0],))
            t_cls = torch.from_numpy(t_cls).long().to(device)

            x_in = torch.cat((patches, x_pos), dim=1)
            logits = net(x_in, t_cls, class_labels=labels, cls_mode=True)

            labels = labels.argmax(dim=1)
            ce_loss = criterion(logits, labels)
            acc = (logits.argmax(dim=1) == labels).float().mean()

            probs = torch.nn.functional.softmax(logits, dim=1)
            ece = ece_criterion.measure(probs.cpu().detach().numpy(), labels.cpu().detach().numpy())

            losses.append(ce_loss.item())
            accs.append(acc.item())
            eces.append(ece)

    net.train()

    return np.mean(losses), np.mean(accs), np.mean(eces)


def initialize(seed: int, cudnn_benchmark: bool):
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
