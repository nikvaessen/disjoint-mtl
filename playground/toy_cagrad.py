from typing import Optional

import torch
import torchvision
import pytorch_lightning


class MtlDataModule(pytorch_lightning.LightningDataModule):

    def __init__(self):
        super().__init__()

        self.train_dl = None
        self.val_dl = None
        self.test_dl = None

    def prepare_data(self) -> None:
        torchvision.datasets.MNIST("./mnist", download=True)
        torchvision.datasets.FashionMNIST("./fashionmnist", download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

class MTLModel(pytorch_lightning.LightningModule):

    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx) :
        pass

    def validation_step(self, batch, batch_idx) :
        pass

    def test_step(self, batch, batch_idx):
        pass


def main():
    dm = MtlDataModule()
    model = MTLModel()

    trainer = pytorch_lightning.Trainer()


if __name__ == '__main__':
    main()
