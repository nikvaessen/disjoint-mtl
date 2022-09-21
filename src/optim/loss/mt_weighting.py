########################################################################################
#
# Implement several weighting techniques for balancing multi-task losses.
#
# Author(s): Nik Vaessen
########################################################################################

from collections import deque
from typing import List, Union

import torch as t
import torch.nn as nn
import torch.nn.functional as F


########################################################################################
# static scaling


class StaticScaling(nn.Module):
    def __init__(self, weights: List[float]):
        super().__init__()

        if sum(weights) != 1:
            raise ValueError(f"weights should sum to 1, but {sum(weights)=}")

        self.weights = nn.Parameter(t.tensor(weights), requires_grad=False)

    def forward(self, *loss_values: t.Tensor):
        if len(loss_values) != len(self.weights):
            raise ValueError(
                f"expected to scale {len(self.weights)=} values, got {len(loss_values)=}"
            )

        # scale each loss value with the static weight
        loss_values = [loss * self.weights[idx] for idx, loss in enumerate(loss_values)]
        weight_values = [self.weights[idx] for idx, _ in enumerate(loss_values)]

        return loss_values, weight_values


########################################################################################
# dynamically scale losses to the lowest or highest value


def _dynamic_scale_generic(*values: t.Tensor, fn: Union[t.min, t.max] = None):
    assert fn is not None
    assert all(len(loss.shape) == 0 for loss in values)
    assert all(v.device == values[0].device for v in values)

    with t.no_grad():
        scalars = t.tensor([*values], device=values[0].device)
        scaled_loss = fn(scalars)
        scale_factors = scaled_loss / scalars

    loss_values = [loss * scale_factors[idx] for idx, loss in enumerate(values)]
    weight_values = [scale_factors[idx] for idx, _ in enumerate(loss_values)]

    return loss_values, weight_values


class DynamicScaling(nn.Module):
    def __init__(self, mode: str):
        super().__init__()

        self.mode = mode

        if self.mode not in ["min", "max"]:
            raise ValueError(f"mode should be either 'min', or 'max', not {mode=}")

    def forward(self, *loss_values: t.Tensor):
        if self.mode == "min":
            return _dynamic_scale_generic(*loss_values, fn=t.min)
        else:
            return _dynamic_scale_generic(*loss_values, fn=t.max)


########################################################################################
# dynamic weight averaging


class DynamicWeightAveraging(nn.Module):
    # implement dynamic weight averaging as described in
    # https://arxiv.org/pdf/1803.10704.pdf

    def __init__(
        self,
        num_losses: int,
        use_average_loss: bool = True,
        average_loss_window: int = 5,
        weight_sum: float = 1.0,
        temperature: float = 2.0,
    ):
        super().__init__()
        self.num_losses = num_losses

        # average
        self.use_average_loss = use_average_loss
        self.average_loss_window = average_loss_window
        if average_loss_window <= 0:
            raise ValueError("average loss window should be at least 1")

        if use_average_loss:
            queue_length = 2 * average_loss_window
        else:
            queue_length = 2

        self.loss_windows = [deque(maxlen=queue_length) for _ in range(self.num_losses)]

        self.weight_sum = nn.Parameter(t.tensor(weight_sum), requires_grad=False)
        self.temperature = nn.Parameter(t.tensor(temperature), requires_grad=False)

    def forward(self, *loss_values: t.Tensor):
        return self._scale(*loss_values)

    def _scale(self, *loss_values: t.Tensor):
        assert len(loss_values) == self.num_losses
        assert all(len(loss.shape) == 0 for loss in loss_values)
        assert all(loss.device == loss_values[0].device for loss in loss_values)

        # determine weighting for each index
        weights = t.tensor(
            [self._weight_k(k) for k in range(self.num_losses)],
            requires_grad=False,
            device=loss_values[0].device,
        )

        # apply temperature, softmax and sum scaling
        weights /= self.temperature
        weights = F.softmax(weights, dim=0)
        weights *= self.weight_sum

        # we store each loss value for further iterations (before they're scaled)
        for idx, loss in enumerate(loss_values):
            self.loss_windows[idx].append(loss.item())

        # scale each loss value with the weight
        loss_values = [loss * weights[idx] for idx, loss in enumerate(loss_values)]
        weight_values = [weights[idx] for idx, _ in enumerate(loss_values)]

        return loss_values, weight_values

    def _weight_k(self, k: int):
        # L_k(t-1)
        numerator = self._loss_k(k, first=True)
        # L_k(t-2)
        denominator = self._loss_k(k, first=False)

        return numerator / denominator

    def _loss_k(self, k: int, first: bool):
        loss_history = list(self.loss_windows[k])

        if len(loss_history) < 2:
            return 1

        middle_idx = len(loss_history) // 2

        if first:
            selected_half = loss_history[middle_idx:]
        else:
            selected_half = loss_history[:middle_idx]

        if self.use_average_loss:
            # take the last n values and average them out
            return sum(selected_half) / len(selected_half)
        else:
            assert len(selected_half) == 1
            return selected_half[0]
