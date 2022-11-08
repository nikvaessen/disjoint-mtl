import torch as t

from fashion_and_mnist.experiment import MTLModel, set_model_weights

path1 = "/home/nik/phd/repo/disjoint_mtl/fashion_and_mnist/resnet18/random_init.ckpt"
path2 = "/home/nik/phd/repo/disjoint_mtl/fashion_and_mnist/resnet18/random_init_n2.ckpt"


model1 = set_model_weights(MTLModel(mode='mnist', model='resnet18'), path1)
model2 = set_model_weights(MTLModel(mode='mnist', model='resnet18'), path2)

model1_param_dict = {k:v for k, v in model1.named_parameters()}
model2_param_dict = {k:v for k, v in model2.named_parameters()}

for k, v1 in model1_param_dict.items():
    v2 = model2_param_dict[k]
    same_elem = t.sum(t.abs(v1 - v2) < 1e-9)
    print(k, same_elem)

print(sum(p.numel() for p in model1.parameters() if p.requires_grad))
