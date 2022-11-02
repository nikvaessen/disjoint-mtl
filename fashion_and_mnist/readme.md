# Experimnets

## STL

### MNIST

```
python run_fashion_mnist.py -m \
hparams.mode=mnist \
hparams.max_lr=1e-6,5e-6,1e-5,5e-5,1e-4,5e-4,1e-3 \
hparams.base_lr_factor=10,50,100,1000 \
hparams.weight_decay=0,1e-1,1e-2,1e-3,1e-4,1e-5 \
hydra/launcher=slurm
```

### FASHIONMNIST

```
python run_fashion_mnist.py -m \
hparams.mode=fashion \
hparams.max_lr=1e-6,5e-6,1e-5,5e-5,1e-4,5e-4,1e-3 \
hparams.base_lr_factor=10,50,100,1000 \
hparams.weight_decay=0,1e-1,1e-2,1e-3,1e-4,1e-5 \
hydra/launcher=slurm
```

## MTL

### Adam

```
python run_fashion_mnist.py -m \
hparams.mode=both \
hparams.max_lr=1e-6,5e-6,1e-5,5e-5,1e-4,5e-4,1e-3 \
hparams.base_lr_factor=10,50,100,1000 \
hparams.weight_decay=0,1e-1,1e-2,1e-3,1e-4,1e-5 \
hydra/launcher=slurm
```

### Adam with ca_grad

```
python run_fashion_mnist.py -m \
hparams.mode=both_cagrad \
hparams.max_lr=1e-6,5e-6,1e-5,5e-5,1e-4,5e-4,1e-3 \
hparams.base_lr_factor=10,50,100,1000 \
hparams.weight_decay=0,1e-1,1e-2,1e-3,1e-4,1e-5 \
hparams.ca_grad_c=0,0.2,0.5,0.8,1 \
hydra/launcher=slurm
```
