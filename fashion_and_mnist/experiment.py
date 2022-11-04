import os
import pathlib

import math
import argparse

from typing import Optional, Dict

import PIL
import torch
import torchmetrics
import torchvision
import pytorch_lightning

import numpy as np
import wandb

from PIL.Image import Image
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.plugins.environments import SLURMEnvironment

from torch import Generator
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataset import T_co
from torchvision.datasets import FashionMNIST, MNIST
from pytorch_lightning.loggers import WandbLogger

from scipy.optimize import minimize_scalar

from src.util.system import get_git_revision_hash


class CombinedDataset(Dataset):
    def __init__(
        self,
        mnist: MNIST,
        fashion_mnist: FashionMNIST,
        mode: str,
        train: bool = True,
        apply_aug: bool = False,
        generator: Generator = torch.Generator().manual_seed(42),
    ):
        assert len(mnist) == len(fashion_mnist)

        self.mnist = mnist
        self.fashion_mnist = fashion_mnist
        self.train = train
        self.mode = mode
        self.apply_aug = apply_aug
        self.generator = generator

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
        self.transform_aug = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
            ]
        )

        if self.mode not in ["mnist", "fashion", "both"]:
            raise ValueError(f"unknown {self.mode=}")

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index) -> T_co:
        if self.train and self.mode == "both":
            fashion_idx = torch.randint(
                low=0, high=len(self.fashion_mnist), size=(), generator=self.generator
            ).item()
        else:
            fashion_idx = index

        transform = self.transform_aug if self.apply_aug else self.transform

        mnist_sample, mnist_gt = self.mnist[index]
        fashion_sample, fashion_gt = self.fashion_mnist[fashion_idx]

        mnist_sample_tensor = transform(mnist_sample)
        fashion_sample_tensor = transform(fashion_sample)
        blank_tensor = torch.zeros_like(mnist_sample_tensor, dtype=torch.float32)

        if self.mode == "both":
            return torch.cat(
                [mnist_sample_tensor, fashion_sample_tensor, blank_tensor], dim=0
            ), (mnist_gt, fashion_gt)
        elif self.mode == "mnist":
            return (
                torch.cat([mnist_sample_tensor, blank_tensor, blank_tensor], dim=0),
                mnist_gt,
            )
        elif self.mode == "fashion":
            return (
                torch.cat([fashion_sample_tensor, blank_tensor, blank_tensor], dim=0),
                fashion_gt,
            )


class MtlDataModule(pytorch_lightning.LightningDataModule):
    def __init__(
        self,
        data_folder: pathlib.Path,
        batch_size: int,
        mode: str = "both",
    ):
        super().__init__()

        self.train_dl = None
        self.val_dl = None
        self.test_dl = None
        self.mode = mode if mode != "both_cagrad" else "both"

        self.batch_size = batch_size

        self.mnist_path = str(data_folder / "mnist")
        self.fashion_mnist_path = str(data_folder / "fashionmnist")

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
        train_ds = CombinedDataset(
            mnist_train, fashion_mnist_train, mode=self.mode, apply_aug=True
        )
        val_ds = CombinedDataset(mnist_val, fashion_mnist_val, mode=self.mode)
        test_ds = CombinedDataset(
            mnist_test, fashion_mnist_test, train=False, mode=self.mode
        )

        self.train_dl = DataLoader(
            train_ds, batch_size=self.batch_size, num_workers=4, shuffle=True
        )
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
        base_lr=1e-4,
        max_lr=1e-3,
        weight_decay=1e-3,
        cycle_steps=400,
        cagrad_c: float = None,
        hparams: Dict = {},
    ):
        super().__init__()

        self.resnet = torchvision.models.resnet18()
        self.resnet.fc = torch.nn.Identity()

        self.fc_mnist = torch.nn.Linear(in_features=512, out_features=10)
        self.fc_fashion = torch.nn.Linear(in_features=512, out_features=10)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.mode = mode
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.weight_decay = weight_decay
        self.cycle_steps = cycle_steps
        self.cagrad_c = cagrad_c

        self.train_acc_mnist = torchmetrics.Accuracy()
        self.train_acc_fashion = torchmetrics.Accuracy()
        self.val_acc_mnist = torchmetrics.Accuracy()
        self.val_acc_fashion = torchmetrics.Accuracy()
        self.test_acc_mnist = torchmetrics.Accuracy()
        self.test_acc_fashion = torchmetrics.Accuracy()

        self.angles = []

        if self.mode == "both_cagrad":
            hparams["cagrad_c"] = cagrad_c
            self.automatic_optimization = False

        self.save_hyperparameters(
            {
                "mode": self.mode,
                "base_lr": self.base_lr,
                "max_lr": max_lr,
                "weight_decay": weight_decay,
                "cycle_steps": cycle_steps,
                **hparams,
            }
        )

    def grad2vec(self, apply_zero_grad: bool = True):
        with torch.no_grad():
            # extract all gradients from all parameters and put them into a single vector
            reconstruction_dict = {}
            stack = []
            start_idx = 0

            for name, param in self.shared_params():
                if param.requires_grad and param.grad is not None:
                    reconstruction_dict[name] = start_idx
                    flat_grad = param.grad.flatten()
                    stack.append(flat_grad)
                    start_idx += flat_grad.shape[0]

            grad_vec = torch.concat(stack)

            if apply_zero_grad:
                self.zero_grad()

            return grad_vec, reconstruction_dict

    def vec2grad(self, vec: torch.Tensor, reconstruction_dict: Dict[str, int]):
        with torch.no_grad():
            # put the single grad vector back into the grad of all parameters
            for name, param in self.shared_params():
                if name in reconstruction_dict:
                    num_params = math.prod(param.shape)
                    begin_idx = reconstruction_dict[name]
                    end_idx = begin_idx + num_params

                    flattened_vec = vec[begin_idx:end_idx]
                    param.grad = flattened_vec.unflatten(-1, param.shape)

    def shared_params(self):
        return self.resnet.named_parameters()

    def training_step(self, batch, batch_idx):
        if self.mode == "both":
            image, (gt_mnist, gt_fashion) = batch

            features = self.resnet(image)

            pred_mnist = self.fc_mnist(features)
            pred_fashion = self.fc_fashion(features)

            loss_mnist = self.loss_fn(pred_mnist, gt_mnist)
            loss_fashion = self.loss_fn(pred_fashion, gt_fashion)

            loss = (loss_fashion + loss_mnist) / 2

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
        elif self.mode == "both_cagrad":
            image, (gt_mnist, gt_fashion) = batch

            opt = self.optimizers()
            sch = self.lr_schedulers()
            opt.zero_grad()

            features = self.resnet(image)

            pred_mnist = self.fc_mnist(features)
            loss_mnist = self.loss_fn(pred_mnist, gt_mnist)
            self.manual_backward(loss_mnist, retain_graph=True)
            g1, g1_info = self.grad2vec()

            pred_fashion = self.fc_fashion(features)
            loss_fashion = self.loss_fn(pred_fashion, gt_fashion)
            self.manual_backward(loss_fashion)
            g2, g2_info = self.grad2vec()

            with torch.no_grad():
                angle = torch.dot(g1, g2)

                self.angles.append(angle.item())
                if len(self.angles) >= 100:
                    num_pos = torch.sum((torch.tensor(self.angles) > 0))
                    self.log("pos_angles_100", num_pos, on_step=True, on_epoch=False)
                    self.angles.clear()

                g0 = (g1 + g2) / 2
                g0_norm = torch.linalg.norm(g0)
                phi = self.cagrad_c**2 * g0_norm**2
                phi_sqrt = torch.sqrt(phi)

                def min_fn(w):
                    w1 = w
                    w2 = 1 - w

                    gw_temp = (w1 * g1) + (w2 * g2) / 2
                    gw_norm_temp = torch.linalg.norm(gw_temp)

                    gwg0 = torch.dot(gw_temp, g0)

                    objective = gwg0 + (phi_sqrt * gw_norm_temp)

                    return objective.cpu().detach().item()

                res = minimize_scalar(min_fn, bounds=(0, 1), method="bounded")
                opt_w1 = res.x
                opt_w2 = 1 - opt_w1

                gw = (opt_w1 * g1) + (opt_w2 * g2)
                gw_norm = torch.linalg.norm(gw)
                coef = phi_sqrt / gw_norm
                g = g0 + (coef * gw)

                self.vec2grad(g, g1_info)
                loss = (loss_fashion + loss_mnist) / 2  # just for logging
                opt.step()
                sch.step()

            self.log("train_loss", loss, on_epoch=False, prog_bar=True)
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
        if self.mode == "both" or self.mode == "both_cagrad":
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
        if self.mode == "both" or self.mode == "both_cagrad":
            image, (gt_mnist, gt_fashion) = batch
        elif self.mode == "mnist":
            image, gt_mnist = batch

        elif self.mode == "fashion":
            image, gt_fashion = batch

        features = self.resnet(image)

        if self.mode == "both" or self.mode == "both_cagrad" or self.mode == "mnist":
            pred_mnist = self.fc_mnist(features)

            self.log(
                "test_mnist_acc",
                self.test_acc_mnist(pred_mnist, gt_mnist),
                prog_bar=True,
            )
        if self.mode == "both" or self.mode == "both_cagrad" or self.mode == "fashion":
            pred_fashion = self.fc_fashion(features)
            self.log(
                "test_fashion_acc",
                self.test_acc_fashion(pred_fashion, gt_fashion),
                prog_bar=True,
            )

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
        sched = CyclicLR(
            opt,
            base_lr=self.base_lr,
            max_lr=self.max_lr,
            step_size_up=self.cycle_steps // 2,
            step_size_down=self.cycle_steps // 2,
            cycle_momentum=False,
            mode="triangular2",
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


def main(
    mode: str = "both",
    batch_size: int = 128,
    cycle_steps: int = 800,
    num_cycles: int = 4,
    max_lr: float = 1e-4,
    base_lr_factor: int = 10,
    weight_decay: float = 1e-3,
    ca_grad_c: float = 0.5,
    use_gpu: bool = True,
    data_folder: pathlib.Path = pathlib.Path("data"),
    experiment_name: str = None,
    tag: str = None,
):
    num_steps = cycle_steps * num_cycles

    dm = MtlDataModule(batch_size=batch_size, mode=mode, data_folder=data_folder)
    model = MTLModel(
        mode=mode,
        base_lr=max_lr / base_lr_factor,
        max_lr=max_lr,
        weight_decay=weight_decay,
        cycle_steps=cycle_steps,
        cagrad_c=ca_grad_c,
        hparams={"max_steps": num_steps, "batch_size": batch_size},
    )

    if experiment_name is None:
        logger = WandbLogger(project="fashion+mnist")
    else:
        logger = WandbLogger(project="fashion+mnist", name=experiment_name, tags=[tag])

    trainer = pytorch_lightning.Trainer(
        logger=logger,
        callbacks=[
            pytorch_lightning.callbacks.LearningRateMonitor(),
            pytorch_lightning.callbacks.ModelCheckpoint(monitor="val_loss"),
        ],
        val_check_interval=cycle_steps // 2,
        check_val_every_n_epoch=None,
        max_steps=num_steps,
        devices=1,
        accelerator="gpu" if use_gpu else "cpu",
        plugins=[SLURMEnvironment(auto_requeue=False)],
    )
    trainer.fit(model, dm)
    trainer.test(model, dm)

    wandb.finish()


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", help="training mode", default="both", type=str)
    parser.add_argument("--batch_size", help="batch_size", default=128, type=int)
    parser.add_argument(
        "--cycle_steps",
        help="number of steps in a single cycle",
        default=1000,
        type=int,
    )
    parser.add_argument("--num_cycles", help="number of cycles", default=5, type=int)
    parser.add_argument(
        "--max_lr", help="maximum LR in cycle", default=1e-4, type=float
    )
    parser.add_argument(
        "--base_lr_factor",
        help="minimum LR in cycle is max_lr/base_lr_factor",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--weight_decay", help="weight decay factor", default=0, type=float
    )
    parser.add_argument(
        "--ca_grad_c",
        help="parameter for ca_grad if mode==both_cagrad",
        default=0,
        type=float,
    )
    args = parser.parse_args()

    main(
        mode=args.mode,
        batch_size=args.batch_size,
        cycle_steps=args.cycle_steps,
        num_cycles=args.num_cycles,
        max_lr=args.max_lr,
        base_lr_factor=args.base_lr_factor,
        weight_decay=args.weight_decay,
        ca_grad_c=args.ca_grad_c,
        use_gpu=True,
    )


def main_from_cfg(cfg: DictConfig):
    # print config
    print(f"current git commit hash: {get_git_revision_hash()}")
    print(f"PyTorch version is {torch.__version__}")
    print(f"PyTorch Lightning version is {pytorch_lightning.__version__}")
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        job_id = os.environ["SLURM_JOB_ID"]
        task_id = os.environ["SLURM_ARRAY_TASK_ID"]
        print(f"detected slurm array job: {job_id}_{task_id}")
    print(OmegaConf.to_yaml(cfg))
    print()

    return main(
        mode=cfg.hparams.mode,
        batch_size=cfg.hparams.batch_size,
        cycle_steps=cfg.hparams.cycle_steps,
        num_cycles=cfg.hparams.num_cycles,
        max_lr=cfg.hparams.max_lr,
        base_lr_factor=cfg.hparams.base_lr_factor,
        weight_decay=cfg.hparams.weight_decay,
        ca_grad_c=cfg.hparams.ca_grad_c,
        data_folder=pathlib.Path(cfg.log_folder),
        use_gpu=True,
        experiment_name=cfg.experiment_name,
        tag=cfg.tag,
    )
