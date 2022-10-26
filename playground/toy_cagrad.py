from typing import Optional

import PIL
import torch
import torchmetrics
import torchvision
import pytorch_lightning

import numpy as np
import wandb

from PIL.Image import Image
from dotenv import load_dotenv

from torch import Generator
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataset import T_co
from torchvision.datasets import FashionMNIST, MNIST
from pytorch_lightning.loggers import WandbLogger


class CombinedDataset(Dataset):
    def __init__(
        self,
        mnist: MNIST,
        fashion_mnist: FashionMNIST,
        mode: str,
        train: bool = True,
        generator: Generator = torch.Generator().manual_seed(42),
    ):
        assert len(mnist) == len(fashion_mnist)

        self.mnist = mnist
        self.fashion_mnist = fashion_mnist
        self.train = train
        self.mode = mode
        self.generator = generator

        if self.mode not in ["mnist", "fashion", "both"]:
            raise ValueError(f"unknown {self.mode=}")

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index) -> T_co:
        if self.train:
            fashion_idx = torch.randint(
                low=0, high=len(self.fashion_mnist), size=(), generator=self.generator
            ).item()
        else:
            fashion_idx = index

        mnist_sample, mnist_gt = self.mnist[index]
        fashion_sample, fashion_gt = self.fashion_mnist[fashion_idx]

        mnist_sample_tensor = torch.tensor(np.array(mnist_sample), dtype=torch.float32)
        fashion_sample_tensor = torch.tensor(
            np.array(fashion_sample), dtype=torch.float32
        )
        blank_tensor = torch.zeros_like(mnist_sample_tensor, dtype=torch.float32)

        if self.mode == "both":
            return torch.stack(
                [mnist_sample_tensor, fashion_sample_tensor, blank_tensor]
            ), (mnist_gt, fashion_gt)
        elif self.mode == "mnist":
            return (
                torch.stack([mnist_sample_tensor, blank_tensor, blank_tensor]),
                mnist_gt,
            )

        elif self.mode == "fashion":
            return (
                torch.stack([fashion_sample_tensor, blank_tensor, blank_tensor]),
                fashion_gt,
            )


def merge_sample(
    sample: Image, other_sample: Image, generator: Generator = None
) -> Image:
    sample_np = np.array(sample)
    other_sample_np = np.array(other_sample)

    width, height = sample_np.shape
    padding = 2
    padding_range = 2 * padding

    merged_np = np.zeros(((width + padding_range) * 2, (height + padding_range) * 2))

    upper_x = 0
    upper_y = 0

    center_x = width + padding_range
    center_y = height + padding_range

    # place sample
    start_x = torch.randint(
        low=upper_x, high=upper_x + padding_range, size=(), generator=generator
    ).item()
    end_x = start_x + width
    start_y = torch.randint(
        low=upper_y, high=upper_y + padding_range, size=(), generator=generator
    ).item()
    end_y = start_y + height

    merged_np[start_x:end_x, start_y:end_y] = sample_np

    # place other sample
    start_x = torch.randint(
        low=center_x,
        high=center_x + padding_range,
        size=(),
        generator=generator,
    ).item()
    end_x = start_x + width
    start_y = torch.randint(
        low=center_y,
        high=center_y + padding_range,
        size=(),
        generator=generator,
    ).item()
    end_y = start_y + height

    merged_np[start_x:end_x, start_y:end_y] = other_sample_np

    # convert back to PIL
    return PIL.Image.fromarray(merged_np)


class MtlDataModule(pytorch_lightning.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        mode: str = "both",
    ):
        super().__init__()

        self.train_dl = None
        self.val_dl = None
        self.test_dl = None
        self.mode = mode

        self.batch_size = batch_size

        self.mnist_path = "./mnist"
        self.fashion_mnist_path = "./fashionmnist"

    def prepare_data(self) -> None:
        MNIST(self.mnist_path, download=True)
        FashionMNIST(self.fashion_mnist_path, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        # split into train/val
        mnist_train, mnist_val = random_split(
            MNIST(self.mnist_path, train=True),
            [50_000, 10_000],
            torch.Generator().manual_seed(42),
        )
        mnist_test = MNIST(self.mnist_path, train=False)

        fashion_mnist_train, fashion_mnist_val = random_split(
            FashionMNIST(self.fashion_mnist_path, train=True),
            [50_000, 10_000],
            torch.Generator().manual_seed(42),
        )
        fashion_mnist_test = FashionMNIST(self.fashion_mnist_path, train=False)

        # create dataloaders
        train_ds = CombinedDataset(mnist_train, fashion_mnist_train, mode=self.mode)
        val_ds = CombinedDataset(mnist_val, fashion_mnist_val, mode=self.mode)
        test_ds = CombinedDataset(
            mnist_test, fashion_mnist_test, train=False, mode=self.mode
        )

        self.train_dl = DataLoader(train_ds, batch_size=self.batch_size, num_workers=1)
        self.val_dl = DataLoader(val_ds, batch_size=self.batch_size, num_workers=1)
        self.test_dl = DataLoader(test_ds, batch_size=self.batch_size, num_workers=1)

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl


class MTLModel(pytorch_lightning.LightningModule):
    def __init__(
        self,
        mode: str,
    ):
        super().__init__()

        self.resnet = torchvision.models.resnet18()
        self.resnet.fc = torch.nn.Identity()

        self.fc_mnist = torch.nn.Linear(in_features=512, out_features=10)
        self.fc_fashion = torch.nn.Linear(in_features=512, out_features=10)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.mode = mode

        self.train_acc_mnist = torchmetrics.Accuracy()
        self.train_acc_fashion = torchmetrics.Accuracy()
        self.val_acc_mnist = torchmetrics.Accuracy()
        self.val_acc_fashion = torchmetrics.Accuracy()
        self.test_acc_mnist = torchmetrics.Accuracy()
        self.test_acc_fashion = torchmetrics.Accuracy()

        self.save_hyperparameters({"mode": self.mode})

    def training_step(self, batch, batch_idx):
        if self.mode == "both":
            image, (gt_mnist, gt_fashion) = batch

            features = self.resnet(image)

            pred_mnist = self.fc_mnist(features)
            pred_fashion = self.fc_fashion(features)

            loss_mnist = self.loss_fn(pred_mnist, gt_mnist)
            loss_fashion = self.loss_fn(pred_fashion, gt_fashion)

            loss = loss_fashion + loss_mnist
            self.log("train_loss", loss, on_epoch=False)
            self.log("loss_fashion", loss_fashion, on_epoch=False)
            self.log("loss_mnist", loss_mnist, on_epoch=False)
            self.log(
                "train_mnist_acc",
                self.train_acc_mnist(pred_mnist, gt_mnist),
                on_epoch=True,
            )
            self.log(
                "train_fashion_acc",
                self.train_acc_fashion(pred_fashion, gt_fashion),
                on_epoch=True,
            )
        elif self.mode == "mnist" or self.mode == "fashion":
            image, gt = batch

            features = self.resnet(image)

            if self.mode == "mnist":
                pred = self.fc_mnist(features)
            else:
                pred = self.fc_fashion(features)

            loss = self.loss_fn(pred, gt)

            self.log("train_loss", loss, on_epoch=False)

            if self.mode == "mnist":
                self.log(
                    "train_mnist_acc",
                    self.train_acc_mnist(pred, gt),
                    on_epoch=True,
                )
            else:
                self.log(
                    "train_fashion_acc",
                    self.train_acc_fashion(pred, gt),
                    on_epoch=True,
                )
        else:
            raise ValueError("")

        return loss

    def validation_step(self, batch, batch_idx):
        if self.mode == "both":
            image, (gt_mnist, gt_fashion) = batch

            features = self.resnet(image)

            pred_mnist = self.fc_mnist(features)
            pred_fashion = self.fc_fashion(features)

            loss_mnist = self.loss_fn(pred_mnist, gt_mnist)
            loss_fashion = self.loss_fn(pred_fashion, gt_fashion)

            self.log(
                "val_mnist_acc", self.val_acc_mnist(pred_mnist, gt_mnist), prog_bar=True
            )
            self.log(
                "val_fashion_acc",
                self.val_acc_fashion(pred_fashion, gt_fashion),
                prog_bar=True,
            )

            loss = loss_fashion + loss_mnist

            self.log("val_loss", loss, prog_bar=True)

        elif self.mode == "mnist" or self.mode == "fashion":
            image, gt = batch

            features = self.resnet(image)

            if self.mode == "mnist":
                pred = self.fc_mnist(features)
            else:
                pred = self.fc_fashion(features)

            loss = self.loss_fn(pred, gt)

            self.log("val_loss", loss, prog_bar=True)

            if self.mode == "mnist":
                self.log(
                    "val_mnist_acc",
                    self.val_acc_mnist(pred, gt),
                    on_epoch=True,
                )
            else:
                self.log(
                    "val_fashion_acc",
                    self.val_acc_fashion(pred, gt),
                    on_epoch=True,
                )
        else:
            raise ValueError("")

    def test_step(self, batch, batch_idx):
        if self.mode == "both":
            image, (gt_mnist, gt_fashion) = batch
        elif self.mode == "mnist":
            image, gt_mnist = batch

        elif self.mode == "fashion":
            image, gt_fashion = batch

        features = self.resnet(image)

        if self.mode == "both" or self.mode == "mnist":
            pred_mnist = self.fc_mnist(features)

            self.log(
                "test_mnist_acc",
                self.test_acc_mnist(pred_mnist, gt_mnist),
                prog_bar=True,
            )
        if self.mode == "both" or self.mode == "fashion":
            pred_fashion = self.fc_fashion(features)
            self.log(
                "test_fashion_acc",
                self.test_acc_fashion(pred_fashion, gt_fashion),
                prog_bar=True,
            )

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=1e-3)
        sched = CyclicLR(
            opt,
            base_lr=1e-5,
            max_lr=1e-3,
            step_size_up=200,
            step_size_down=200,
            cycle_momentum=False,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


def main(mode="mnist"):
    dm = MtlDataModule(batch_size=128, mode=mode)
    model = MTLModel(mode=mode)

    trainer = pytorch_lightning.Trainer(
        logger=WandbLogger(project="fashion+mnist"),
        callbacks=[
            pytorch_lightning.callbacks.LearningRateMonitor(),
            pytorch_lightning.callbacks.ModelCheckpoint(monitor="val_loss"),
        ],
        val_check_interval=100,
        check_val_every_n_epoch=None,
        max_steps=2400,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, dm)
    trainer.test(model, dm)

    wandb.finish()


if __name__ == "__main__":
    load_dotenv()
    main(mode="both")
    main(mode="mnist")
    main(mode="fashion")
