## Speaker recognition

### wav2vec2 with vox2, ls960h, ls_clean, ls_other

```bash
python run_speaker.py -m +experiments=speaker_aam_4cycle_nofreeze_ch3s_bs64 \
network=speaker_wav2vec2_linear \
data/module=speaker_voxceleb,speaker_librispeech_960h,speaker_librispeech_clean,speaker_librispeech_other \
optim.algo.lr=1e-4,1.78e-4 \
hydra/launcher=slurm hydra.launcher.array_parallelism=4 hydra.launcher.timeout_min=1440
```

experiment with chunk size and batch size

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
