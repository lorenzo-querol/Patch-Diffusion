import csv
import os
import random
import time
import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from calibration.ece import ECELoss
import dnnlib


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, in_channels=3, num_classes=10, dropRate=0.3):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = BasicBlock

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(in_channels, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def get_cifar10_loaders(batch_size=128, num_workers=4):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load datasets
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return {"train": train_loader, "test": test_loader}, 10


def train_epoch(epoch, model, train_loader, criterion, optimizer, scheduler, device):
    model.train()

    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    ece_meter = AverageMeter()

    ece_criterion = ECELoss()

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, dim=1)

        loss_ = loss.item()
        correct_ = preds.eq(targets).sum().item()
        num = inputs.size(0)
        ece = ece_criterion(outputs, targets).item()

        accuracy = correct_ / num

        loss_meter.update(loss_, num)
        acc_meter.update(accuracy, num)
        ece_meter.update(ece, num)

    scheduler.step()

    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
        "ece": ece_meter.avg,
    }


def evaluate(model, test_loader, criterion, device):
    model.eval()

    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    ece_meter = AverageMeter()

    ece_criterion = ECELoss(10)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, preds = torch.max(outputs, dim=1)

            loss_ = loss.item()
            correct_ = preds.eq(targets).sum().item()
            num = inputs.size(0)
            ece = ece_criterion(outputs, targets).item()

            accuracy = correct_ / num

            loss_meter.update(loss_, num)
            acc_meter.update(accuracy, num)
            ece_meter.update(ece, num)

    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
        "ece": ece_meter.avg,
    }


def train_model(model, train_loader, test_loader, num_epochs=200, run_dir="wrn-runs", snapshot_freq=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    logger = dnnlib.util.Logger(file_name=os.path.join(run_dir, "log.txt"), file_mode="a", should_flush=True)
    metrics_file = os.path.join(run_dir, "metrics.csv")
    with open(metrics_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["epoch", "train_loss", "train_acc", "train_ece", "test_loss", "test_acc", "test_ece"])
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        time_start = time.time()
        train_metrics = train_epoch(epoch, model, train_loader, criterion, optimizer, scheduler, device)
        elapsed = time.time() - time_start
        test_metrics = evaluate(model, test_loader, criterion, device)

        logger.write(
            f"Epoch {epoch}/{num_epochs} done in {elapsed:.0f}s\t"
            f"Train loss: {train_metrics['loss']:.4f}, acc: {train_metrics['acc']:.4f}, ECE: {train_metrics['ece']:.4f}\t\t"
            f"Test loss: {test_metrics['loss']:.4f}, acc: {test_metrics['acc']:.4f}, ECE: {test_metrics['ece']:.4f}\n"
        )

        with open(metrics_file, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["epoch", "train_loss", "train_acc", "train_ece", "test_loss", "test_acc", "test_ece"])
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["acc"],
                    "train_ece": train_metrics["ece"],
                    "test_loss": test_metrics["loss"],
                    "test_acc": test_metrics["acc"],
                    "test_ece": test_metrics["ece"],
                }
            )

        if epoch % snapshot_freq == 0:
            torch.save(model.state_dict(), os.path.join(run_dir, f"network-snapshot-{epoch}.pt"))


def get_next_run_dir(base_dir, desc="run"):
    # If directory doesn't exist, start with run 0
    os.makedirs(base_dir, exist_ok=True)

    # Get all numeric prefixes of existing directories
    existing_runs = [int(d[:5]) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d[:5].isdigit()]

    # Get next run number
    next_run = max(existing_runs, default=-1) + 1

    # Create directory
    name = f"{next_run:05d}-{desc}"
    os.makedirs(os.path.join(base_dir, name), exist_ok=False)

    return os.path.join(base_dir, name)


def init_environment(seed):
    if seed is None:
        seed = random.randint(0, 2**31)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True


@click.command()
@click.option("--batch_size", default=128, help="Batch size")
@click.option("--num_epochs", default=200, help="Number of epochs")
@click.option("--run_dir", default="wrn-runs", help="Directory to save run metrics")
@click.option("--seed", default=None, help="Seed for reproducibility")
@click.option("--snapshot_freq", default=10, help="Frequency of saving network")
@click.option("--test", is_flag=True, help="Run in test mode")
@click.option("--network_path", default=None, help="Path to network snapshot")
@click.option("--test_dir", default="test-wrn-runs", help="Directory to save test metrics")

# WRN-related
@click.option("--depth", default=28, help="Depth of WideResNet")
@click.option("--widen_factor", default=10, help="Widen factor of WideResNet")
def main(**kwargs):
    args = dnnlib.EasyDict(kwargs)
    init_environment(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders, num_classes = get_cifar10_loaders(batch_size=args.batch_size)
    model = WideResNet(depth=args.depth, widen_factor=args.widen_factor, num_classes=num_classes)

    if args.test:
        assert args.network_path is not None, "Please provide a network path for testing"

        model.load_state_dict(torch.load(args.network_path))
        model = model.to(device)

        test_loader = loaders["test"]
        criterion = nn.CrossEntropyLoss()

        test_metrics = evaluate(model, test_loader, criterion, device)

        run_name = os.path.dirname(args.network_path).split("/")[-1]
        os.makedirs(args.test_dir, exist_ok=True)

        with open(os.path.join(args.test_dir, f"{run_name}.csv"), "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["loss", "acc", "ece"])
            writer.writeheader()
            writer.writerow(
                {
                    "loss": test_metrics["loss"],
                    "acc": test_metrics["acc"],
                    "ece": test_metrics["ece"],
                }
            )

        return

    run_dir = get_next_run_dir(args.run_dir)
    train_model(model, loaders["train"], loaders["test"], num_epochs=args.num_epochs, run_dir=run_dir, snapshot_freq=args.snapshot_freq)


if __name__ == "__main__":
    """
    python train_wrn.py --test --network_path wrn-runs/00003-run/network-snapshot-200.pt
    """
    main()
