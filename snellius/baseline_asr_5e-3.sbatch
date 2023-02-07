#!/bin/bash

#SBATCH --array=0-3%2
#SBATCH --cpus-per-task=16
#SBATCH --error=/home/nvaessen/logs/%A_%a_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=run_speech
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nvaessen
#SBATCH --mem=120GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/nvaessen/logs/%A_%a_0_log.out
#SBATCH --partition=gpu
#SBATCH --time=2880

if [ "$SLURM_ARRAY_TASK_ID" = 0 ]
then
python run_speech.py -m \
data/module=speech_ls960h \
network=speech_wav2vec2_linear \
optim.algo.lr=3e-5 \
tag=asr_ls \
fi

if [ "$SLURM_ARRAY_TASK_ID" = 1 ]
then
python run_speech.py -m \
data/module=speech_ls960h \
network=speech_wav2vec2_linear_no_reg \
optim.algo.lr=3e-5 \
tag=asr_ls \
fi

if [ "$SLURM_ARRAY_TASK_ID" = 2 ]
then
python run_speech.py -m \
data/module=speech_vox2 \
network=speech_wav2vec2_linear \
optim.algo.lr=3e-5 \
tag=asr_vox \
fi

if [ "$SLURM_ARRAY_TASK_ID" = 3 ]
then
python run_speech.py -m \
data/module=speech_vox2 \
network=speech_wav2vec2_linear_no_reg \
optim.algo.lr=3e-5 \
tag=asr_vox \
fi
