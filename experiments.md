
```bash
python run_speaker.py -m +experiments=ls_speaker_960 \
network=speaker_wavlm_fc,speaker_wav2vec2_fc \
optim.algo.lr=1e-4,1e-5 \
data.speaker_datapipe.train_dp.batch_size=64,128 \
hydra/launcher=slurm_24vram hydra.launcher.array_parallelism=8
```