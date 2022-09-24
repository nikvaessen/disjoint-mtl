########################################################################################
#
# Utility around freezing a module
#
# Author(s): Nik Vaessen
########################################################################################

from typing import Optional, Union, List

import torch as t

########################################################################################
# manages whether a torch module should be frozen


class FreezeManager:
    def __init__(
        self,
        module: Union[t.nn.Module, List[t.nn.Module]],
        is_frozen_at_init: bool,
        num_steps_frozen: Optional[int] = None,
    ):
        if isinstance(module, t.nn.Module):
            self.module = [module]
        else:
            self.module = module

        self.is_frozen_at_init = is_frozen_at_init
        self.num_steps_frozen = num_steps_frozen

        self._num_steps = 0
        self._should_unfreeze = False

    def on_train_start(self) -> None:
        if self.is_frozen_at_init:
            for m in self.module:
                freeze_module(m)

            self._should_unfreeze = (
                self.num_steps_frozen is not None and self.num_steps_frozen > 0
            )

        self._num_steps = 0

    def on_after_backward(self) -> None:
        if self._should_unfreeze:
            self._num_steps += 1

            if self._num_steps > self.num_steps_frozen:
                for m in self.module:
                    unfreeze_module(m)
                self._should_unfreeze = False


########################################################################################
# freeze/unfreeze a module and all it's parameters


def freeze_module(mod: t.nn.Module):
    mod.requires_grad_(False)


def unfreeze_module(mod: t.nn.Module):
    mod.requires_grad_(True)
