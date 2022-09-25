########################################################################################
#
# This file is the main entrypoint of the train/eval loop based on the
# hydra configuration.
#
# Author(s): Nik Vaessen
########################################################################################

import logging

from typing import List, Dict, Union, Callable

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

import src.optim.loss.cross_entropy
from data_utility.pipe.builder import (
    SpeakerRecognitionDataPipeBuilder,
    SpeechRecognitionDataPipeBuilder,
)
from src.data.module.librispeech import (
    LibriSpeechDataModuleConfig,
    LibriSpeechDataModule,
)
from src.networks.wav2vec2.w2v2_speaker import (
    Wav2vec2ForSpeakerRecognitionConfig,
    Wav2vec2ForSpeakerRecognition,
)
from src.networks.wav2vec2.w2v2_speech import (
    Wav2vec2ForSpeechRecognitionConfig,
    Wav2vec2ForSpeechRecognition,
)
from src.networks.wavlm.wavlm_speaker import (
    WavLMForSpeakerRecognitionConfig,
    WavLMForSpeakerRecognition,
)
from src.networks.wavlm.wavlm_speech import WavLMForSpeechRecognitionConfig, WavLMForSpeechRecognition
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


def construct_speaker_recognition_module(
    cfg: DictConfig,
    network_cfg: Union[
        Wav2vec2ForSpeakerRecognitionConfig, WavLMForSpeakerRecognitionConfig
    ],
    dm: Union[LibriSpeechDataModule],
    loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
):
    # every speaker recognition network needs to be given these variables
    # for training purposes
    num_speakers = dm.get_num_train_speakers()
    test_pairs = dm.get_test_speaker_eval_list()
    test_names = dm.get_test_names()

    # get init function based on config type
    if isinstance(network_cfg, Wav2vec2ForSpeakerRecognitionConfig):
        network_class = Wav2vec2ForSpeakerRecognition
    elif isinstance(network_cfg, WavLMForSpeakerRecognitionConfig):
        network_class = WavLMForSpeakerRecognition
    else:
        raise ValueError(f"cannot load network from {network_cfg}")

    # init model
    kwargs = {
        "root_hydra_config": cfg,
        "loss_fn_constructor": loss_fn_constructor,
        "num_speakers": num_speakers,
        "test_pairs": test_pairs,
        "test_names": test_names,
        "cfg": network_cfg,
    }

    return init_model(cfg, network_class, kwargs)


def construct_speech_recognition_module(
    cfg: DictConfig,
    network_cfg: Union[
        Wav2vec2ForSpeechRecognitionConfig, WavLMForSpeechRecognitionConfig
    ],
    dm: Union[LibriSpeechDataModule],
    loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
):
    # every speaker recognition network needs to be given these variables
    # for training purposes
    test_names = dm.get_test_names()
    idx_to_char = dm.get_idx_to_char()

    # get init function based on config type
    if isinstance(network_cfg, Wav2vec2ForSpeechRecognitionConfig):
        network_class = Wav2vec2ForSpeechRecognition
    elif isinstance(network_cfg, WavLMForSpeechRecognitionConfig):
        network_class = WavLMForSpeechRecognition
    else:
        raise ValueError(f"cannot load network from {network_cfg}")

    # init model
    kwargs = {
        "root_hydra_config": cfg,
        "loss_fn_constructor": loss_fn_constructor,
        "idx_to_char": idx_to_char,
        "test_names": test_names,
        "cfg": network_cfg,
    }

    return init_model(cfg, network_class, kwargs)


def init_model(cfg: DictConfig, network_class, kwargs: Dict):
    # load model weights from checkpoint
    potential_checkpoint_path = cfg.get("load_network_from_checkpoint", None)

    if potential_checkpoint_path is not None:
        log.info(
            f"reloading {network_class.__class__} from {potential_checkpoint_path}"
        )
        network = network_class.load_from_checkpoint(
            cfg.load_network_from_checkpoint, strict=False, **kwargs
        )
    else:
        network = network_class(**kwargs)

    return network


def construct_network_module(
    cfg: DictConfig,
    dm: Union[LibriSpeechDataModule],
):
    # load loss function
    def loss_fn_constructor():
        # should be instantiated in the network
        # so that potential parameters are properly
        # registered
        return instantiate(cfg.optim.loss)

    # load network config
    network_cfg = instantiate(cfg.network)

    if isinstance(network_cfg, Wav2vec2ForSpeakerRecognitionConfig):
        network = construct_speaker_recognition_module(
            cfg, network_cfg, dm, loss_fn_constructor
        )
    elif isinstance(network_cfg, WavLMForSpeakerRecognitionConfig):
        network = construct_speaker_recognition_module(
            cfg, network_cfg, dm, loss_fn_constructor
        )
    elif isinstance(network_cfg, Wav2vec2ForSpeechRecognitionConfig):
        network = construct_speech_recognition_module(
            cfg, network_cfg, dm, loss_fn_constructor
        )
    elif isinstance(network_cfg, WavLMForSpeechRecognitionConfig):
        network = construct_speech_recognition_module(
            cfg, network_cfg, dm, loss_fn_constructor
        )
    else:
        raise ValueError(
            f"can not construct network for network_cfg type {network_cfg.__class__}"
        )

    # set optimizer and learning rate schedule
    optimizer = instantiate(cfg.optim.algo, params=network.parameters())
    schedule = {
        "scheduler": instantiate(cfg.optim.schedule.scheduler, optimizer=optimizer),
        "monitor": cfg.optim.schedule.monitor,
        "interval": cfg.optim.schedule.interval,
        "frequency": cfg.optim.schedule.frequency,
        "name": cfg.optim.schedule.name,
    }
    # remove None values from dict
    schedule = {k: v for k, v in schedule.items() if v is not None}

    network.set_optimizer(optimizer)
    network.set_lr_schedule(schedule)

    return network


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

    # train model
    if cfg.fit_model:
        trainer.fit(network, datamodule=dm)

    # test model
    if cfg.trainer.accelerator == "ddp":
        destroy_process_group()

        if not trainer.global_rank == 0:
            return

        trainer: pl.Trainer = instantiate(
            cfg.trainer,
            devices=min(1, cfg.trainer.get("devices", 0)),
            logger=logger,
            callbacks=callbacks,
            profiler=profiler,
        )

    if cfg.eval_model and cfg.fit_model:
        # this will select the checkpoint with the best validation metric
        # according to the ModelCheckpoint callback
        try:
            trainer.test(datamodule=dm)
        except:
            # there might not have been a validation epoch
            trainer.test(network, datamodule=dm)
    elif cfg.eval_model:
        # this will simply test the given model weights (when it's e.g
        # manually loaded from a checkpoint)
        trainer.test(network, datamodule=dm)

    return None
