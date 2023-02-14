################################################################################
#
# This callback implements visualization of the tensor(s) which are
# directly put into the model.
#
# It will:
#   1. print the raw input and ground truth tensor to a file
#   2. print some statistics of the input tensor to a file
#   3. try to convert the tensor back to a wav file and save it to disk
#
# Author(s): Nik Vaessen
################################################################################

import logging
from collections import defaultdict

from typing import Any

import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import Optimizer

################################################################################
# callback implementation

# A logger for this file
log = logging.getLogger(__name__)


class MagnitudeMonitor(Callback):
    def __init__(self, frequency: int = 100):
        self.cache = defaultdict(list)
        self.count = 0
        self.log_frequency = frequency
        assert self.log_frequency >= 0

    def on_before_optimizer_step(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        if self.log_frequency < 1:
            return

        self.count += 1
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                weight_min = torch.min(param).to("cpu")
                weight_max = torch.max(param).to("cpu")
                weight_avg = torch.mean(param).to("cpu")

                if (
                    param.requires_grad
                    and param.grad is not None
                    and not torch.isnan(param.grad).any()
                    and not torch.isinf(param.grad).any()
                ):
                    grad_min = torch.min(param.grad).to("cpu")
                    grad_max = torch.max(param.grad).to("cpu")
                    grad_avg = torch.mean(param.grad).to("cpu")
                else:
                    grad_min = torch.tensor(torch.nan)
                    grad_max = torch.tensor(torch.nan)
                    grad_avg = torch.tensor(torch.nan)

                stack_tensor = torch.stack(
                    [weight_min, weight_avg, weight_max, grad_min, grad_avg, grad_max]
                )
                self.cache[name].append(stack_tensor)

            if self.count >= self.log_frequency:
                values_list = []

                for v in self.cache.values():
                    values_list.extend(v)

                self.count = 0
                self.cache.clear()

                values = torch.stack(values_list)
                values_min = torch.nan_to_num(values, nan=torch.inf)
                values_max = torch.nan_to_num(values, nan=-torch.inf)

                weight_avg = torch.mean(values[:, 1])
                grad_avg = torch.nanmean(values[:, 4])

                weight_min = torch.min(values_min[:, 0])
                grad_min = torch.min(values_min[:, 3])

                weight_max = torch.max(values_max[:, 2])
                grad_max = torch.max(values_max[:, 5])

                pl_module.log_dict(
                    {
                        f"weight_avg": weight_avg,
                        f"weight_min": weight_min,
                        f"weight_max": weight_max,
                        f"grad_avg": grad_avg,
                        f"grad_min": grad_min,
                        f"grad_max": grad_max,
                    }
                )
