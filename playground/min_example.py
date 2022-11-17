import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split

from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from pytorch_lightning.plugins.environments import SLURMEnvironment


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train, val = random_split(dataset, [55000, 5000])

    autoencoder = LitAutoEncoder()
    trainer = pl.Trainer(
        devices=1, accelerator="gpu", plugins=SLURMEnvironment(auto_requeue=False)
    )
    trainer.fit(autoencoder, DataLoader(train), DataLoader(val))


if __name__ == "__main__":
    main()
