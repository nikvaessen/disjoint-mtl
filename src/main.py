########################################################################################
#
# This file is the main entrypoint of the train/eval loop based on the
# hydra configuration.
#
# Author(s): Nik Vaessen
########################################################################################

import logging

from typing import List

import torch as t
import pytorch_lightning as pl
import transformers
import wandb

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.distributed import destroy_process_group
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_utility.pipe.builder import (
    SpeakerRecognitionDataPipeBuilder,
    SpeechRecognitionDataPipeBuilder,
)
from src.data.module.librispeech import (
    LibriSpeechDataModuleConfig,
    LibriSpeechDataModule,
)
from src.util.system import get_git_revision_hash

log = logging.getLogger(__name__)


########################################################################################
# implement constructing data module


def construct_speech_data_pipe_builders(cfg: DictConfig):
    train_dp = SpeechRecognitionDataPipeBuilder(
        cfg=instantiate(cfg.data.speech_datapipe.train_dp)
    )
    val_dp = SpeechRecognitionDataPipeBuilder(
        cfg=instantiate(cfg.data.speech_datapipe.val_dp)
    )
    test_dp = SpeechRecognitionDataPipeBuilder(
        cfg=instantiate(cfg.data.speech_datapipe.test_dp)
    )

    return train_dp, val_dp, test_dp


def construct_speaker_data_pipe_builders(cfg: DictConfig):
    train_dp = SpeakerRecognitionDataPipeBuilder(
        cfg=instantiate(cfg.data.speaker_datapipe.train_dp)
    )
    val_dp = SpeakerRecognitionDataPipeBuilder(
        cfg=instantiate(cfg.data.speaker_datapipe.val_dp)
    )
    test_dp = SpeakerRecognitionDataPipeBuilder(
        cfg=instantiate(cfg.data.speaker_datapipe.test_dp)
    )

    return train_dp, val_dp, test_dp


def construct_data_module(cfg: DictConfig):
    dm_cfg = instantiate(cfg.data.module)

    speech_dpb = construct_speech_data_pipe_builders(cfg)
    speaker_dpb = construct_speaker_data_pipe_builders(cfg)

    if isinstance(dm_cfg, LibriSpeechDataModuleConfig):
        dm = LibriSpeechDataModule(
            dm_cfg, speech_pipe_builders=speech_dpb, speaker_pipe_builders=speaker_dpb
        )
    else:
        raise ValueError(f"no suitable constructor for {dm_cfg}")

    return dm


########################################################################################
# implement construction of network modules


def construct_network_module(cfg: DictConfig):
    pass


########################################################################################
# implement construction of callbacks, profiler and logger


def construct_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks = []

    callback_cfg: DictConfig = cfg.callbacks

    ModelCheckpoint.CHECKPOINT_NAME_LAST = callback_cfg.get(
        "last_checkpoint_pattern", "last"
    )

    for cb_key in callback_cfg.to_add:
        if cb_key is None:
            continue

        if cb_key in callback_cfg:
            cb = instantiate(callback_cfg[cb_key])
            log.info(f"Using callback <{cb}>")

            callbacks.append(instantiate(callback_cfg[cb_key]))

    return callbacks


def construct_profiler(cfg: DictConfig):
    profile_cfg = cfg.get("profiler", None)

    if profile_cfg is None:
        return None
    else:
        return instantiate(profile_cfg)


def construct_logger(cfg: DictConfig):
    if cfg.use_wandb:
        if isinstance(cfg.tag, str):
            cfg.tag = [cfg.tag]
        if cfg.run_lr_range_test:
            cfg.tag.append("lr_range_test")

        logger = WandbLogger(
            project=cfg.project_name,
            name=cfg.experiment_name,
            tags=cfg.tag,
        )
        # init the wandb agent
        _ = logger.experiment
    else:
        logger = True

    return logger


########################################################################################
# implement the main function based on the whole config


def run_train_eval_script(cfg: DictConfig):
    # create logger
    logger = construct_logger(cfg)

    # print config
    print(OmegaConf.to_yaml(cfg))
    print(f"current git commit hash: {get_git_revision_hash()}")
    print(f"PyTorch version is {t.__version__}")
    print(f"PyTorch Lightning version is {pl.__version__}")
    print(f"transformers version is {transformers.__version__}")
    print()

    # construct data module
    dm = construct_data_module(cfg)

    # create callbacks
    callbacks = construct_callbacks(cfg)

    # construct profiler
    profiler = construct_profiler(cfg)

    # create training/evaluator
    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        profiler=profiler,
    )

    # construct lighting module for train/test
    network = construct_network_module(cfg, dm)

    exit()

    # train model
    if cfg.fit_model:
        trainer.fit(network, datamodule=dm)

    # test model
    if cfg.trainer.accelerator == "ddp":
        destroy_process_group()

        if not trainer.global_rank == 0:
            return

    # create a new trainer which uses at most 1 gpu
    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        gpus=min(1, int(cfg.trainer.get("gpus"))),
        accelerator=None,
        logger=logger,
        callbacks=callbacks,
        profiler=profiler,
    )

    result = None
    if cfg.eval_model and cfg.fit_model:
        # this will select the checkpoint with the best validation metric
        # according to the ModelCheckpoint callback
        try:
            result = trainer.test(datamodule=dm)
        except:
            # there might not have been a validation epoch
            result = trainer.test(network, datamodule=dm)
    elif cfg.eval_model:
        # this will simply test the given model weights (when it's e.g
        # manually loaded from a checkpoint)
        result = trainer.test(network, datamodule=dm)

    if result is not None:
        if isinstance(result, list):
            result_obj = result[0]

            if "test_eer_val" in result_obj:
                objective = result_obj["test_eer_val"]
            elif "test_eer_o" in result_obj:
                objective = result_obj["test_eer_o"]
            elif "test_wer_other" in result_obj:
                objective = result_obj["test_wer_other"]
            else:
                raise ValueError(
                    f"unknown objective value out of keys "
                    f"{[k for k in result_obj.keys()]} and {result}"
                )

            return objective
        else:
            raise ValueError(f"result object has unknown type {type(result)=}")

    return None
