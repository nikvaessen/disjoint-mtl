
### SV on 960h librispeech with wavLM and wav2vec2

```bash
python run_speaker.py -m +experiments=ls_960_speaker_aam \
network=speaker_wav2vec2_linear,speaker_wavlm_linear \
optim.algo.lr=1e-4,1e-5 \
data.speaker_datapipe.train_dp.batch_size=64,128 \
hydra/launcher=slurm_24vram hydra.launcher.array_parallelism=8 \
```

### SV on disjoint librispeech with wavLM and wav2vec2

```bash
python run_speaker.py -m +experiments=ls_disjoint_speaker_aam \
network=speaker_wav2vec2_linear,speaker_wavlm_linear \
optim.algo.lr=1e-4,1e-5 \
data.speaker_datapipe.train_dp.batch_size=64 \
hydra.launcher.array_parallelism=8
```

### ASR on 960h librispeech with wavLM and wav2vec2


```bash
python run_speech.py -m +experiments=ls_speech_960 \
network=speech_wav2vec2_linear,speech_wavvlm_linear \
optim.algo.lr=1e-4,5e-5,1e-5 \
callbacks=speech_early_stopping \
hydra/launcher=slurm_24vram hydra.launcher.array_parallelism=6 \
hydra.launcher.timeout_min=4800
```

### SV on voxceleb with wav2vec2 and wavLM, different LR and chunk size 1, 2, 3

```
python run_speaker.py -m +experiments=vox_speaker \
network=speaker_wavlm_linear,speaker_wav2vec2_linear \
optim.algo.lr=1e-5,1e-4 \
data.speaker_datapipe.train_dp.chunk_size_sec=1,2,3 
hydra/launcher=slurm_11vram hydra.launcher.array_parallelism=2 \
hydra.launcher.timeout_min=1440
```