## Speaker recognition

### wav2vec2 with vox2, ls960h, ls_clean, ls_other

```bash
python run_speaker.py -m +experiments=speaker_aam_4cycle_nofreeze_ch3s_bs64 \
network=speaker_wav2vec2_linear \
data/module=speaker_voxceleb,speaker_librispeech_960h,speaker_librispeech_clean,speaker_librispeech_other \
optim.algo.lr=1e-4,1.78e-4 \
hydra/launcher=slurm hydra.launcher.array_parallelism=4 hydra.launcher.timeout_min=1440
```

### experiment with chunk size and batch size

```bash
python run_speaker.py -m +experiments=speaker_aam_4cycle_nofreeze_ch3s_bs64 \
data/module=speaker_voxceleb \
network=speaker_wav2vec2_linear \
data.speaker_datapipe.train_dp.batch_size=192 \
data.speaker_datapipe.train_dp.chunk_size_sec=1 \
hydra/launcher=slurm hydra.launcher.timeout_min=1440
```

```bash
python run_speaker.py -m +experiments=speaker_aam_4cycle_nofreeze_ch3s_bs64 \
data/module=speaker_voxceleb \
network=speaker_wav2vec2_linear \
data.speaker_datapipe.train_dp.batch_size=64 \
data.speaker_datapipe.train_dp.chunk_size_sec=1 \
trainer.max_steps=200_000 \
hydra/launcher=slurm hydra.launcher.timeout_min=1440
```

### experiment with chunking within wav2vec2

```
python run_speaker.py -m +experiments=speaker_aam_4cycle_nofreeze_ch3s_bs64,speaker_ce_4cycle_nofreeze_ch3s_bs64 \
data/pipe=speaker_datapipe_whole \
network.head_cfg.train_random_chunk_size=40,120 \
network.head_cfg.enable_train_chunk=true \
hydra/launcher=slurm_24vram hydra.launcher.array_parallelism=4 hydra.launcher.timeout_min=1440
```

### experiments with different heads

```
python run_speaker.py -m +experiments=speaker_aam_4cycle_nofreeze_ch3s_bs64 \
network=speaker_wav2vec2_linear network.head_cfg.use_projection_layer=false \
project_name=speakers_heads \
hydra/launcher=slurm hydra.launcher.array_parallelism=4 hydra.launcher.timeout_min=1440
```

```
python run_speaker.py -m +experiments=speaker_aam_4cycle_nofreeze_ch3s_bs64 \
network=speaker_wav2vec2_linear network.head_cfg.use_projection_layer=true \
network.head_cfg.projection_layer_dim=64,128,256,512,1024 \
project_name=speakers_heads \
hydra/launcher=slurm hydra.launcher.array_parallelism=5 hydra.launcher.timeout_min=1440
```

```
python run_speaker.py -m +experiments=speaker_aam_4cycle_nofreeze_ch3s_bs64 \
network=speaker_wav2vec2_xvector,speaker_wav2vec2_ecapa \
project_name=speakers_heads \
hydra/launcher=slurm hydra.launcher.array_parallelism=4 hydra.launcher.timeout_min=1440
```


## ASR

### wav2vec2 with ls960, ls_clean, ls_other

```bash
python run_speech.py -m +experiments=speech_ctc_3st_freeze\
data/module=speech_librispeech_960h,speech_librispeech_clean,speech_librispeech_other \
network=speech_wav2vec2_linear optim.algo.lr=1e-4 \
hydra/launcher=slurm_24vram hydra.launcher.array_parallelism=3 \
hydra.launcher.timeout_min=4800
```
### wavLM with ls960, ls_clean, ls_other

```bash
python run_speech.py -m +experiments=speech_ctc_3st_freeze\
data/module=speech_librispeech_960h,speech_librispeech_clean,speech_librispeech_other \
network=speech_wavlm_linear optim.algo.lr=1e-5 \
hydra/launcher=slurm_24vram hydra.launcher.array_parallelism=3 \
hydra.launcher.timeout_min=4800
```

## MTL (joined)

### wav2vec2 librispeech ls960

```
python run_mt_speech_speaker.py -m +experiments=mtl_librispeech_only \
trainer.max_steps=100_000,320_000 \
hydra/launcher=slurm_24vram hydra.launcher.array_parallelism=2 \
hydra.launcher.timeout_min=4800
```

with chunking speaker head

```
python run_mt_speech_speaker.py -m +experiments=mtl_librispeech_only \
trainer.max_steps=100_000 \
network.speaker_head_cfg.train_random_chunk_size=20,40,80,120 \
network.speaker_head_cfg.enable_train_chunk=true \
hydra/launcher=slurm_24vram hydra.launcher.array_parallelism=4 \
hydra.launcher.timeout_min=4800
```

# log

### exp 1

```
python run_speaker.py -m +experiments=speaker_aam_4cycle_nofreeze_ch3s_bs64 network=speaker_wav2vec2_linear \
network.head_cfg.pool_method=mean,first data.pipe.speaker.train_dp.chunk_size_sec=3,9 tag=exp1 \
hydra/launcher=slurm_24vram optim.algo.lr=0.0001,5e-5,1e-5 hydra.launcher.array_parallelism=2
```

### exp 2

```
python run_mtl_disjoint.py -m +experiments=mtl_vox2_ls optim.algo.lr=1e-4,5e-5,1e-5 \
network.speaker_head_cfg.pool_method=first data/module=mtl_disjoint_ls960h,mtl_disjoint_ls960h_vox2 \
data.pipe.speaker.train_dp.chunk_size_sec=3,9 tag=exp2 hydra/launcher=slurm_24vram 
```

### exp 3

python run_mtl_disjoint.py -m +experiments=mtl_vox2_ls_dsi data.pipe.speaker.train_dp.chunk_size_sec=3 optim.loss.scale_method=dwa optim.algo.lr=5e-5 hydra/launcher=slurm_24vram tag=dsi network.dsi_head_alpha=1e-3,1e-4,1e-5 tag=exp3
