########################################################################################
#
# This file is the main entrypoint of the train/eval loop based on the
# hydra configuration.
#
# Author(s): Nik Vaessen
########################################################################################

import logging

from typing import Union, Callable, List, Dict

import torch as t
import pytorch_lightning as pl
import transformers
import wandb

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.distributed import destroy_process_group
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger, WandbLogger

from src.util.system import get_git_revision_hash

log = logging.getLogger(__name__)


########################################################################################
# implement constructing data module


########################################################################################
# implement the main function based on the whole config


def run_train_eval_script(cfg: DictConfig):
    if cfg.anonymous_mode:
        import warnings

        # warnings leak absolute path of python files (and thus username)
        warnings.filterwarnings("ignore")

        # pytorch lightning might log absolute path of checkpoint files, and thus
        # leak username
        logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)

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
        auto_lr_find="auto_lr_find" if cfg.run_lr_range_test else None,
    )

    # construct lighting module for train/test
    network = construct_network_module(cfg, dm)

    # tune model
    if cfg.run_lr_range_test:
        trainer.logger.log_hyperparams(cfg)
        run_lr_range_test(
            trainer,
            network,
            dm,
            tune_iterations=cfg.lr_range_iterations,
        )

        return

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
