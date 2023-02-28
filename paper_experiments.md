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

train

```
poetry run python run_speech.py -m \
data/module=speech_ls960h \
network=speech_wav2vec2_linear,speech_wav2vec2_linear_no_reg \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
tag=asr_ls \
data.pipe.speech.train_dp.num_workers=12 \
hydra/launcher=das_preempt
```

eval

```
poetry run python run_speech.py -m \
data/module=speech_eval \
network=speech_wav2vec2_linear \
fit_model=False \
load_network_from_checkpoint=~/repo/disjoint_mtl/models/stl_asr_ls_social-benches-7.ckpt \
tag=eval \
hydra/launcher=das_preempt
```

#### vox2

train
```
poetry run python run_speech.py -m \
data/module=speech_vox2 \
network=speech_wav2vec2_linear,speech_wav2vec2_linear_no_reg \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
tag=asr_vox \
data.pipe.speech.train_dp.num_workers=12 \
hydra/launcher=das_preempt
```

eval
```
poetry run python run_speech.py -m \
data/module=speech_eval \
network=speech_wav2vec2_linear \
fit_model=False \
load_network_from_checkpoint=~/repo/disjoint_mtl/models/stl_asr_v2_surrounding-symptons-5.ckpt \
tag=eval \
hydra/launcher=das_preempt
```

### SKR

#### librispeech

train
```
poetry run python run_speaker.py -m \
data/module=speaker_ls960h \
network=speaker_wav2vec2_linear,speaker_wav2vec2_linear_no_reg \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
tag=skr_ls \
data.pipe.speaker.train_dp.num_workers=12 \
hydra/launcher=das_preempt
```

eval
```
poetry run python run_speaker.py -m \
data/module=speaker_eval \
network=speaker_wav2vec2_linear \
fit_model=False \
load_network_from_checkpoint=~/repo/disjoint_mtl/models/stl_skr_ls_damaging-readers-7.ckpt \
data.pipe.speaker.test_dp.chunk_strategy=start \
data.pipe.speaker.test_dp.chunk_size_sec=2,10,2_000_000_000 \
data.module.speaker_json='${oc.env:LIBRISPEECH_DIR}/meta/speakers_train.json'
tag=eval \
hydra/launcher=das_preempt  hydra.launcher.array_parallelism=1
```

#### vox2

train
```
poetry run python run_speaker.py -m \
data/module=speaker_vox2 \
network=speaker_wav2vec2_linear,speaker_wav2vec2_linear_no_reg \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
tag=skr_vox \
data.pipe.speaker.train_dp.num_workers=12 \
hydra/launcher=das_preempt
```

eval
```
poetry run python run_speaker.py -m \
data/module=speaker_eval \
network=speaker_wav2vec2_linear \
fit_model=False \
load_network_from_checkpoint=~/repo/disjoint_mtl/models/stl_skr_v2_explicit-rhythms-6.ckpt \
data.pipe.speaker.test_dp.chunk_strategy=start \
data.pipe.speaker.test_dp.chunk_size_sec=2,10,2_000_000_000 \
tag=eval \
hydra/launcher=das_preempt  hydra.launcher.array_parallelism=1
```

## MTL

### joint

#### librispeech

train
```
poetry run python run_mtl_joint.py -m \
data/module=mtl_joint_ls960h \
network=mtl_joint_wav2vec2_linear,mtl_joint_wav2vec2_linear_no_reg \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
tag=mtl_j_ls \
hydra/launcher=icis_preempt
```

eval
```
poetry run python run_mtl_disjoint.py -m \
data/module=mtl_disjoint_eval \
network=mtl_disjoint_wav2vec2_linear \
fit_model=False \
load_network_from_checkpoint=~/repo/disjoint_mtl/models/mtl_joint_ls_angry-attendances-7.ckpt \
data.pipe.speaker.test_dp.chunk_strategy=start \
data.pipe.speaker.test_dp.chunk_size_sec=2,10,2_000_000_000 \
data.module.speaker_dm_cfg.speaker_json='${oc.env:LIBRISPEECH_DIR}/meta/speakers_train.json' \
tag=eval \
hydra/launcher=snellius
```

#### librispeech + voxceleb

train
```
poetry run python run_mtl_joint.py -m \
data/module=mtl_joint_ls960h_vox2 \
network=mtl_joint_wav2vec2_linear,mtl_joint_wav2vec2_linear_no_reg \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
tag=mtl_j_ls_vox \
hydra/launcher=icis_preempt
```
```
poetry run python run_mtl_joint.py -m \
data/module=mtl_joint_ls960h_vox2 \
network=mtl_joint_wav2vec2_linear \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
optim.loss.static_speech_weight=0.9 optim.loss.static_speaker_weight=0.1 \
tag=mtl_j_ls_vox \
hydra/launcher=icis_preempt
```


eval
```
poetry run python run_mtl_disjoint.py -m \
data/module=mtl_disjoint_eval \
network=mtl_disjoint_wav2vec2_linear \
fit_model=False \
load_network_from_checkpoint=~/repo/disjoint_mtl/models/mtl_joint_ls+v2_normal-buses-9.ckpt \
data.pipe.speaker.test_dp.chunk_strategy=start \
data.pipe.speaker.test_dp.chunk_size_sec=2,10,2_000_000_000 \
data.module.speaker_dm_cfg.speaker_json='${oc.env:LIBRISPEECH_DIR}/meta/vox2_ls960h_speakers.json' \
tag=eval \
hydra/launcher=snellius
```

### disjoint, 2 seconds

#### librispeech

train
```
poetry run python run_mtl_disjoint.py -m \
data/module=mtl_disjoint_ls960h \
network=mtl_disjoint_wav2vec2_linear,mtl_disjoint_wav2vec2_linear_no_reg \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
tag=mtl_dj_ls \
hydra/launcher=icis_preempt
```

eval
```
poetry run python run_mtl_disjoint.py -m \
data/module=mtl_disjoint_eval \
network=mtl_disjoint_wav2vec2_linear \
fit_model=False \
load_network_from_checkpoint=~/repo/disjoint_mtl/models/mtl_dj_2s_ls_small-opposites-0.ckpt \
data.pipe.speaker.test_dp.chunk_strategy=start \
data.pipe.speaker.test_dp.chunk_size_sec=2,10,2_000_000_000 \
data.module.speaker_dm_cfg.speaker_json='${oc.env:LIBRISPEECH_DIR}/meta/speakers_train.json' \
tag=eval \
hydra/launcher=snellius
```


#### librispeech + voxceleb

train
```
poetry run python run_mtl_disjoint.py -m \
data/module=mtl_disjoint_ls960h_vox2 \
network=mtl_disjoint_wav2vec2_linear,mtl_disjoint_wav2vec2_linear_no_reg \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
tag=mtl_dj_ls_vox \
hydra/launcher=icis_preempt
```
```
poetry run python run_mtl_disjoint.py -m \
data/module=mtl_disjoint_ls960h_vox2 \
network=mtl_disjoint_wav2vec2_linear \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
optim.loss.static_speech_weight=0.9 optim.loss.static_speaker_weight=0.1 \
tag=mtl_dj_ls_vox \
hydra/launcher=icis_preempt
```

eval
```
poetry run python run_mtl_disjoint.py -m \
data/module=mtl_disjoint_eval \
network=mtl_disjoint_wav2vec2_linear \
fit_model=False \
load_network_from_checkpoint=~/repo/disjoint_mtl/models/mtl_dj_2s_ls+v2_silly-homelands-7.ckpt \
data.pipe.speaker.test_dp.chunk_strategy=start \
data.pipe.speaker.test_dp.chunk_size_sec=2,10,2_000_000_000 \
tag=eval \
hydra/launcher=snellius
```

### disjoint, 10 seconds

#### librispeech

train
```
poetry run python run_mtl_disjoint.py -m \
data/module=mtl_disjoint_ls960h \
network=mtl_disjoint_wav2vec2_linear,mtl_disjoint_wav2vec2_linear_no_reg \
data.pipe.speaker.train_dp.chunk_size_sec=10 \
data.pipe.speaker.train_dp.batch_size=20 \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
tag=mtl_dj10_ls \
hydra/launcher=icis_preempt
```

eval
tbd

#### librispeech + voxceleb

train
```
poetry run python run_mtl_disjoint.py -m \
data/module=mtl_disjoint_ls960h_vox2 \
network=mtl_disjoint_wav2vec2_linear,mtl_disjoint_wav2vec2_linear_no_reg \
data.pipe.speaker.train_dp.chunk_size_sec=10 \
data.pipe.speaker.train_dp.batch_size=20 \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
tag=mtl_dj10_ls_vox \
hydra/launcher=icis_preempt
```
```
poetry run python run_mtl_disjoint.py -m \
data/module=mtl_disjoint_ls960h_vox2 \
network=mtl_disjoint_wav2vec2_linear \
data.pipe.speaker.train_dp.chunk_size_sec=10 \
data.pipe.speaker.train_dp.batch_size=20 \
optim.algo.lr=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4 \
optim.loss.static_speech_weight=0.9 optim.loss.static_speaker_weight=0.1 \
tag=mtl_dj10_ls_vox \
hydra/launcher=icis_preempt
```

eval
```
poetry run python run_mtl_disjoint.py -m \
data/module=mtl_disjoint_eval \
network=mtl_disjoint_wav2vec2_linear \
fit_model=False \
load_network_from_checkpoint=~/repo/disjoint_mtl/models/mtl_dj_10s_ls+v2_passionate-capitalism-0.ckpt \
data.pipe.speaker.test_dp.chunk_strategy=start \
data.pipe.speaker.test_dp.chunk_size_sec=2,10,2_000_000_000 \
tag=eval \
hydra/launcher=snellius
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
-e 'ssh -J bastion-user@bastion-host:22'
# Table 3

all model checkpoints result from experiments in Table 1


