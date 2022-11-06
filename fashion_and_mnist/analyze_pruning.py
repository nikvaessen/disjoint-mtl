import matplotlib
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from matplotlib import pyplot as plt

from experiment import MTLModel


def prune_model(model, factor):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, "weight", factor)


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


prune_percentage = []
overlap_percentage = []

for i in range(1, 100):
    model = MTLModel(model="resnet152", mode='mnist')
    model.load_state_dict(
        torch.load(
            "/home/nik/phd/repo/disjoint_mtl/fashion_and_mnist/resnet152/random_init_n1.ckpt"
        ).state_dict()#['state_dict']
    )

    # model_mnist = MTLModel.load_from_checkpoint(
    #     "/home/nik/phd/repo/disjoint_mtl/fashion_and_mnist/resnet152/mnist_final.ckpt", model="resnet152"
    # ).cuda()
    # model_fashion = MTLModel.load_from_checkpoint(
    #     "resnet152/fashion_final.ckpt",  model="resnet152"
    # ).cuda()
    print(i)
    factor = i / 100
    prune_model(model_mnist, factor)
    prune_model(model_fashion, factor)
    overlap = compare_masks(model_mnist, model_fashion)

    prune_percentage.append(factor)
    overlap_percentage.append(overlap)

plt.plot(prune_percentage, overlap_percentage)
plt.show()
