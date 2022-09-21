################################################################################
#
# Define a base lightning module for speech and/or speaker recognition network.
#
# Author(s): Nik Vaessen
################################################################################

import logging

from typing import Callable, Optional

import torch as t

import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf

################################################################################
# Definition of speaker recognition API

# A logger for this file

log = logging.getLogger(__name__)


class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        root_hydra_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
    ):
        super().__init__()

        # input arguments
        self.loss_fn = loss_fn_constructor()

        # created by set_methods
        self.optimizer = None
        self.schedule = None
        self.warmup_optimizer = None
        self.warmup_schedule = None

        # log hyperparameters
        self.save_hyperparameters(OmegaConf.to_container(root_hydra_config))

    def set_optimizer(self, optimizer: t.optim.Optimizer):
        self.optimizer = optimizer

    def set_lr_schedule(self, schedule: t.optim.lr_scheduler._LRScheduler):
        self.schedule = schedule

    def configure_optimizers(self):
        return [self.optimizer], [self.schedule]
