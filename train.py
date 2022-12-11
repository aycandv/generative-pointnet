import os
import torch
import argparse
import logging
import wandb

from tqdm import tqdm
from termcolor import colored
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader

from models.pointnet import PointNet
from utils.dataset import PointCloudData
from utils.transforms import (
    train_transforms,
    PointSampler,
    Normalize,
    RandRotation_z,
    RandomNoise,
    ToTensor,
)
from models.generate_model import generate_model
from metrics.generate_loss import generate_loss


wandb.init(project="generative-pointnet", entity="kevi")


def train(args):
    # Create dataset
    train_dataset = PointCloudData(
        args.root_dir, folder="train", transform=train_transforms
    )
    valid_dataset = PointCloudData(
        args.root_dir, valid=True, folder="test", transform=train_transforms
    )

    # Create dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    # Create model
    model = generate_model(args.model, args.device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create loss function
    criterion = generate_loss(type=args.model)

    best_acc = 0
    best_model = model

    wandb.watch(model, criterion, log="all", log_graph=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # pbar
        pbar = tqdm(
            train_loader,
            total=len(train_loader),
            leave=True,
            unit="batch",
            ascii=True,
            position=0,
            desc="Epoch {:4d}".format(epoch + 1),
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}], {rate_fmt} {postfix}",
        )
        for i, data in enumerate(pbar):
            inputs, labels = data["pointcloud"].to(args.device).float(), data[
                "category"
            ].to(args.device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = model(inputs.transpose(1, 2))

            loss = criterion(outputs, inputs, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            # _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # correct += (predicted == labels).sum().item()

            running_loss += loss.item()
            # pbar.set_postfix({'loss': '%.3f' % (running_loss / 10), 'acc': '%.3f' % (100 * correct / total)})

            if i % 10 == 9:  # print every 10 mini-batches
                pbar.set_postfix(
                    {
                        "loss": "%.3f" % (running_loss / total),
                    }
                )
                running_loss = 0.0

        wandb.log(
            {"train/loss": loss.item(), "train/acc": 100 * correct / len(train_dataset)}
        )

        model.eval()
        correct = total = 0
        running_loss = 0.0

        # validation
        if valid_loader:
            pbar_val = tqdm(
                valid_loader,
                total=len(valid_loader),
                leave=True,
                unit="batch",
                ascii=True,
                position=0,
                desc="Validation",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}], {rate_fmt}{postfix}",
            )
            with torch.no_grad():
                for data in pbar_val:
                    inputs, labels = data["pointcloud"].to(args.device).float(), data[
                        "category"
                    ].to(args.device)
                    outputs, m3x3, m64x64 = model(inputs.transpose(1, 2))
                    # _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    # correct += (predicted == labels).sum().item()
                    loss = criterion(outputs, inputs, m3x3, m64x64)
                    running_loss += loss.item()
                    # pbar_val.set_postfix(loss=running_loss /total , acc=correct/total)
            val_acc = 100.0 * correct / total

            pbar_val.set_postfix(
                loss=running_loss / len(valid_loader), acc=correct / total
            )
            wandb.log(
                {"val/acc": val_acc, "val/loss": running_loss / len(valid_loader)}
            )

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), args.save_dir + "/best_model.pth")
            # log best model info with green color
            logging.info(
                colored(
                    f"Best model saved at epoch {epoch+1} with acc {best_acc}", "green"
                )
            )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--root_dir", type=str, default="data/ModelNet10", help="path to dataset"
    )
    args.add_argument("--batch_size", type=int, default=4, help="input batch size")
    args.add_argument("--num_workers", type=int, default=0, help="number of workers")
    args.add_argument(
        "--epochs", type=int, default=6, help="number of epochs to train for"
    )
    args.add_argument("--lr", type=float, default=0.001, help="learning rate")
    args.add_argument("--momentum", type=float, default=0.9, help="momentum")
    args.add_argument("--weight_decay", type=float, default=0.0005, help="weight decay")
    args.add_argument("--device", type=str, default="cuda", help="device to use")
    args.add_argument(
        "--save_dir", type=str, default="checkpoints", help="path to save model"
    )
    args.add_argument("--model", type=str, default="pointnet", help="model to use")
    args = args.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # log hyperparameters
    logging.info(args)

    wandb.config.update(args)

    train(args)
