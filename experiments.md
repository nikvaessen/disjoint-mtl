## Speaker recognition

### vox2 train data

```bash
python run_speaker.py -m +experiments=speaker_aam_4cycle_nofreeze_ch1s_bs64 \
data/module=voxceleb \
network=speaker_wav2vec2_linear,speaker_wavlm_linear \
optim.algo.lr=1e-4 \
data.speaker_datapipe.train_dp.batch_size=64 \
data.speaker_datapipe.train_dp.chunk_size_sec=1,3  \
hydra/launcher=slurm hydra.launcher.array_parallelism=2 hydra.launcher.timeout_min=1440 \
tag=['vox2']
```

### librispeech data

```bash
python run_speaker.py -m +experiments=speaker_aam_4cycle_nofreeze_ch1s_bs64 \
data/module=librispeech_speaker \
network=speaker_wav2vec2_linear,speaker_wavlm_linear \
optim.algo.lr=1e-4 \
data.speaker_datapipe.train_dp.batch_size=64 \
data.speaker_datapipe.train_dp.chunk_size_sec=1,3  \
hydra/launcher=slurm hydra.launcher.array_parallelism=2 hydra.launcher.timeout_min=1440 \
tag=['ls960h']
```

### disjoint librispeech data

TODO

## ASR

###

```bash
python run_speech.py -m +experiments=speech_ctc_3st_freeze \
network=speech_wav2vec2_linear,speech_wavvlm_linear \
optim.algo.lr=1e-4,5e-5,1e-5 \
callbacks=speech_early_stopping \
hydra/launcher=slurm_24vram hydra.launcher.array_parallelism=6 \
hydra.launcher.timeout_min=4800 \
tag=['ls960h']
```
