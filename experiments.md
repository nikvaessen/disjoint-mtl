## Speaker recognition

### vox2 train data

```bash
python run_speaker.py -m +experiments=speaker_aam_4cycle_nofreeze_ch1s_bs64 \
data/module=speaker_voxceleb \
network=speaker_wav2vec2_linear,speaker_wavlm_linear \
optim.algo.lr=1.78e-4 \
data.speaker_datapipe.train_dp.batch_size=64 \
data.speaker_datapipe.train_dp.chunk_size_sec=1,3  \
hydra/launcher=slurm hydra.launcher.array_parallelism=2 hydra.launcher.timeout_min=1440 \
tag=['vox2']
```

### LibriSpeech 960h 

```bash
python run_speaker.py -m +experiments=speaker_aam_4cycle_nofreeze_ch1s_bs64 \
data/module=speaker_librispeech_960h \
network=speaker_wav2vec2_linear,speaker_wavlm_linear \
optim.algo.lr=1.78e-4 \
data.speaker_datapipe.train_dp.batch_size=64 \
data.speaker_datapipe.train_dp.chunk_size_sec=1,3  \
hydra/launcher=slurm hydra.launcher.array_parallelism=2 hydra.launcher.timeout_min=1440 \
tag=['ls960h']
```

### disjoint LibriSpeech

#### clean

```bash
python run_speaker.py -m +experiments=speaker_aam_4cycle_nofreeze_ch1s_bs64 \
data/module=speaker_librispeech_clean \
network=speaker_wav2vec2_linear,speaker_wavlm_linear \
optim.algo.lr=1.78e-4 \
data.speaker_datapipe.train_dp.batch_size=64 \
data.speaker_datapipe.train_dp.chunk_size_sec=1,3  \
hydra/launcher=slurm hydra.launcher.array_parallelism=2 hydra.launcher.timeout_min=1440 \
tag=['ls_clean']
```

#### other

```bash
python run_speaker.py -m +experiments=speaker_aam_4cycle_nofreeze_ch1s_bs64 \
data/module=speaker_librispeech_other \
network=speaker_wav2vec2_linear,speaker_wavlm_linear \
optim.algo.lr=1.78e-4 \
data.speaker_datapipe.train_dp.batch_size=64 \
data.speaker_datapipe.train_dp.chunk_size_sec=1,3  \
hydra/launcher=slurm hydra.launcher.array_parallelism=2 hydra.launcher.timeout_min=1440 \
tag=['ls_other']
```

## ASR

### LibriSpeech 960h 

```bash
python run_speech.py -m +experiments=speech_ctc_3st_freeze \
data/module=speech_librispeech_960h \
network=speech_wav2vec2_linear,speech_wavvlm_linear \
optim.algo.lr=1e-4 \
hydra/launcher=slurm_24vram hydra.launcher.array_parallelism=2 \
hydra.launcher.timeout_min=4800 \
tag=['ls960h']
```

### disjoint LibriSpeech

#### clean

```bash
python run_speech.py -m +experiments=speech_ctc_3st_freeze \
data/module=speech_librispeech_clean \
network=speech_wav2vec2_linear,speech_wavvlm_linear \
optim.algo.lr=1e-4 \
hydra/launcher=slurm_24vram hydra.launcher.array_parallelism=2 \
hydra.launcher.timeout_min=4800 \
tag=['ls_clean']
```

#### other

```bash
python run_speech.py -m +experiments=speech_ctc_3st_freeze \
data/module=speech_librispeech_other \
network=speech_wav2vec2_linear,speech_wavvlm_linear \
optim.algo.lr=1e-4 \
hydra/launcher=slurm_24vram hydra.launcher.array_parallelism=2 \
hydra.launcher.timeout_min=4800 \
tag=['ls_other']
```