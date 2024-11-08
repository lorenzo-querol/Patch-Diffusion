import click
import pickle
import torch
from calibration.ece import ECELoss
import dnnlib
import os
import numpy as np

from tqdm import tqdm


def eval_cls(net, loss_fn, dataloader, resolution, device):
    losses, accs, eces = [], [], []
    num_classes = 10  # Assuming CIFAR-10; adjust if using a different dataset

    # Initialize counters for per-class accuracy
    correct_per_class = torch.zeros(num_classes).to(device)
    total_per_class = torch.zeros(num_classes).to(device)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device).to(torch.float32) / 127.5 - 1
            labels = labels.to(device)

            logits, ce_loss = loss_fn(
                net=net,
                images=images,
                labels=labels,
                patch_size=resolution,
                resolution=resolution,
                cls_mode=True,
            )

            # Get predicted and true classes
            predicted = logits.argmax(dim=1)
            true_labels = labels.argmax(dim=1)

            # Update overall metrics
            acc = (predicted == true_labels).float().mean()
            ece = ECELoss()(logits, true_labels)

            # Update per-class accuracy counters
            for class_idx in range(num_classes):
                mask = true_labels == class_idx
                if mask.sum() > 0:  # if there are any samples of this class
                    correct_per_class[class_idx] += (predicted[mask] == true_labels[mask]).sum()
                    total_per_class[class_idx] += mask.sum()

            losses.append(ce_loss.mean().item())
            accs.append(acc.item())
            eces.append(ece.item())

    # Calculate per-class accuracies
    per_class_acc = torch.zeros(num_classes)
    for i in range(num_classes):
        if total_per_class[i] > 0:
            per_class_acc[i] = (correct_per_class[i] / total_per_class[i]).cpu()

    # Print per-class accuracies
    print("\nPer-class accuracies:")
    for i in range(num_classes):
        print(f"Class {i}: {per_class_acc[i]*100:.2f}% ({int(correct_per_class[i])}/{int(total_per_class[i])})")

    return np.mean(losses), np.mean(accs), np.mean(eces), per_class_acc.numpy()



@click.command()
@click.option("--network", "network_pkl", help="Network pickle filename", metavar="PATH|URL", type=str, required=True)
@click.option("--outdir", help="Where to save the results", metavar="DIR", type=str, required=True)
@click.option("--test_dir", help="Path to the test dataset", metavar="ZIP|DIR", type=str, required=True)
def main(
    network_pkl,
    test_dir,
    outdir,
    device=torch.device("cuda"),
):
    # To run this script, use the following command:
    # python test_cls.py --network <network_pkl> --outdir <outdir> --test_dir <test_dir>
    # Example:
    # python test_cls.py --network training-runs/00000-train-cond-ebm-pedm-gpus2-batch128-fp32/network-snapshot-000000.pkl --outdir results --test_dir ../data/cifar10/test.zip
    # python test_cls.py --network training-runs/00000-train-cond-ebm-pedm-gpus2-batch128-fp32/network-snapshot-015072.pkl --outdir results --test_dir ../data/cifar10/test.zip

    # Load the model.
    with open(network_pkl, "rb") as f:
        net = pickle.load(f)["ema"].to(device)
    net.eval()

    loss_kwargs = {"class_name": "training.patch_loss.Patch_EDMLoss"}
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)  # training.loss.(VP|VE|EDM)Loss

    c = dnnlib.EasyDict()

    # Load the test data.
    c.test_dataset_kwargs = dnnlib.EasyDict(
        class_name="training.dataset.ImageFolderDataset",
        path=test_dir,
        use_labels=True,
        xflip=False,
        cache=True,
    )
    c.data_loader_kwargs = dnnlib.EasyDict(
        batch_size=256,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=2,
    )
    test_dataset_obj = dnnlib.util.construct_class_by_name(**c.test_dataset_kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset_obj, **c.data_loader_kwargs)

    # Create the output directory.
    os.makedirs(outdir, exist_ok=True)

    # test_loss, test_acc, test_ece = eval_cls(net, loss_fn, test_dataloader, test_dataset_obj.resolution, device)

    # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test ECE: {test_ece:.4f}")
    test_loss, test_acc, test_ece, per_class_acc = eval_cls(net, loss_fn, test_dataloader, test_dataset_obj.resolution, device)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test ECE: {test_ece:.4f}")

    # Optionally save the results
    results = {"test_loss": test_loss, "test_acc": test_acc, "test_ece": test_ece, "per_class_acc": per_class_acc}


if __name__ == "__main__":
    main()
