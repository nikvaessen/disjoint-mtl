import logging
from pathlib import Path

import pandas as pd
import torch

import seaborn as sns
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer

from experiment import MTLModel, MtlDataModule, prune_model
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def collect_masks(model):
    mask_dict = {}

    for name, buffer in model.named_buffers():
        if "mask" in name:
            mask_dict[name] = buffer

    return mask_dict


def compare_masks(model, other_model):
    masks = collect_masks(model)
    other_masks = collect_masks(other_model)

    total_numel = 0
    total_overlap = 0

    with torch.no_grad():
        for name, mask in masks.items():
            other_mask = other_masks[name]

            overlap = mask == other_mask
            overlap_count = torch.sum(overlap).item()
            total = torch.numel(mask)

            total_numel += total
            total_overlap += overlap_count

    return total_overlap / total_numel


def eval_model(model, dm):
    trainer = Trainer(devices=1, accelerator="gpu", enable_progress_bar=False)
    result = trainer.test(model, dm, verbose=False)

    if len(result) > 1:
        raise ValueError

    if "test_mnist_acc" in result[0]:
        return result[0]["test_mnist_acc"]

    elif "test_fashion_acc" in result[0]:
        return result[0]["test_fashion_acc"]

    else:
        print(result)
        raise NotImplemented()


model = []
prune_percentage = []
overlap_percentage = []
mnist_acc = []
fashion_acc = []

data_folder = "/home/nik/phd/repo/disjoint_mtl/data/logs/disjoint_mtl/fashion+mnist/"
dm_mnist = MtlDataModule(
    data_folder=Path(data_folder),
    batch_size=256,
    mode="mnist",
)
dm_fashion = MtlDataModule(
    data_folder=Path(data_folder),
    batch_size=256,
    mode="fashion",
)

ckpt_dict = {
    "resnet18": {
        "mnist": "/home/nik/phd/repo/disjoint_mtl/fashion_and_mnist/resnet18/mnist_final.ckpt",
        "fashion": "/home/nik/phd/repo/disjoint_mtl/fashion_and_mnist/resnet18/fashion_final.ckpt",
    },
    "resnet152": {
        "mnist": "/home/nik/phd/repo/disjoint_mtl/fashion_and_mnist/resnet152/mnist_final.ckpt",
        "fashion": "/home/nik/phd/repo/disjoint_mtl/fashion_and_mnist/resnet152/fashion_final.ckpt",
    },
}

for model_version in ["resnet18", "resnet152"]:
    for i in range(0, 99, 1):
        print(model_version, i)

        model_mnist = MTLModel.load_from_checkpoint(
            ckpt_dict[model_version]["mnist"],
            model=model_version,
        ).cuda()
        model_fashion = MTLModel.load_from_checkpoint(
            ckpt_dict[model_version]["fashion"],
            model=model_version,
        ).cuda()

        factor = i / 100
        prune_model(model_mnist, factor)
        prune_model(model_fashion, factor)

        overlap = compare_masks(model_mnist, model_fashion)

        model.append(model_version)
        prune_percentage.append(factor)
        overlap_percentage.append(overlap)
        mnist_acc.append(eval_model(model_mnist, dm_mnist))
        fashion_acc.append(eval_model(model_fashion, dm_fashion))

df = pd.DataFrame(
    {
        "model": model,
        "prune_percentage": prune_percentage,
        "overlap_percentage": overlap_percentage,
        "mnist_acc": mnist_acc,
        "fashion_acc": fashion_acc,
    }
)

print(df)

plt.rcParams["figure.figsize"] = (20, 10)
fix, axes = plt.subplots(1, 3)

sns.lineplot(df, x="prune_percentage", y="overlap_percentage", hue="model", ax=axes[0])
sns.lineplot(df, x="prune_percentage", y="mnist_acc", hue="model", ax=axes[1])
sns.lineplot(df, x="prune_percentage", y="fashion_acc", hue="model", ax=axes[2])

plt.savefig("resnet18_vs_resnet152.png")
plt.show()
