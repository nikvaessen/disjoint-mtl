# general hparams

train 200k steps
regularisation = default huggingface, none
LR = {1, 5.62} x 10-6, {1, 5.62} x 10−5, {1, 5.62} x 10−4 
batch size = 3.2M samples
tri-stage LR (10%, 40%, 50%)
for ASR, set bias of FC layer to prior of letters, freeze transformers 3k steps
freeze cnn during train

pick best checkpoint during training based on 1/4 val WER + 1/4 val EER
pick best LR based on 1/4 dev WER + dev EER

report eval metrics:
test set ASR: librispeech test-other, switchboard?
test set SKR: vox2-hard, switchboard?

loss functions:
CTC-loss for ASR
AAM-softmax loss for SKR
MTL: lambda={0.5, 0.9} static weighting between CTC loss and AAM-softmax loss (no dwa)

# Table 1

wav2vec2 BASE, 1 fc for ASR, mean-pooling + 1 FC for SKR

## STL

### ASR librispeech

#### librispeech

```
python run_speech.py -m \
data/module=speech_ls960h \
network=speech_wav2vec2_linear,speech_wav2vec2_linear_no_reg \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
tag=asr_ls \
hydra/launcher=slurm_24vram
```

#### vox2

```
python run_speech.py -m \
data/module=speech_vox2 \
network=speech_wav2vec2_linear,speech_wav2vec2_linear_no_reg \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
tag=asr_vox \
hydra/launcher=slurm_24vram
```

### SKR

#### librispeech

```
python run_speaker.py -m \
data/module=speaker_ls960h \
network=speaker_wav2vec2_linear,speaker_wav2vec2_linear_no_reg \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
tag=skr_ls \
hydra/launcher=slurm_24vram
```

#### vox2

```
python run_speaker.py -m \
data/module=speaker_vox2 \
network=speaker_wav2vec2_linear,speaker_wav2vec2_linear_no_reg \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
tag=skr_vox \
hydra/launcher=slurm_24vram
```

## MTL

### joint

#### librispeech

```
python run_mtl_joint.py -m \
data/module=mtl_joint_ls960h \
network=mtl_joint_wav2vec2_linear,mtl_joint_wav2vec2_linear_no_reg \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
tag=mtl_j_ls \
hydra/launcher=slurm_24vram
```

#### librispeech + voxceleb

```
python run_mtl_joint.py -m \
data/module=mtl_joint_ls960h_vox2 \
network=mtl_joint_wav2vec2_linear,mtl_joint_wav2vec2_linear_no_reg \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
tag=mtl_j_ls_vox \
hydra/launcher=slurm_24vram
```

### disjoint, 2 seconds

#### librispeech

```
python run_mtl_disjoint.py -m \
data/module=mtl_disjoint_ls960h \
network=mtl_disjoint_wav2vec2_linear,mtl_disjoint_wav2vec2_linear_no_reg \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
tag=mtl_dj_ls_vox \
hydra/launcher=slurm_24vram
```

#### librispeech + voxceleb

```
python run_mtl_disjoint.py \
data/module=mtl_disjoint_ls960h_vox2 \
network=mtl_disjoint_wav2vec2_linear,mtl_disjoint_wav2vec2_linear_no_reg \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
tag=mtl_dj_ls_vox \
hydra/launcher=slurm_24vram
```

### disjoint, 10 seconds

#### librispeech

```
python run_mtl_disjoint.py -m \
data/module=mtl_disjoint_ls960h \
network=mtl_disjoint_wav2vec2_linear,mtl_disjoint_wav2vec2_linear_no_reg \
data.pipe.speaker.train_dp.chunk_size_sec=10 \
data.pipe.speaker.train_dp.batch_size=20 \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
tag=mtl_dj10_ls \
hydra/launcher=slurm_24vram
```

#### librispeech + voxceleb

```
python run_mtl_disjoint.py -m \
data/module=mtl_disjoint_ls960h_vox2 \
network=mtl_disjoint_wav2vec2_linear,mtl_disjoint_wav2vec2_linear_no_reg \
data.pipe.speaker.train_dp.chunk_size_sec=10 \
data.pipe.speaker.train_dp.batch_size=20 \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
tag=mtl_dj10_ls_vox \
hydra/launcher=slurm_24vram
```

# Table 2

## STL - 3 architectures
data=vox2

```
TODO
```

## MTL disjoint:
data=ls+vox2

```
TODO
```

# Table 3

all model checkpoints result from experiments in Table 1


