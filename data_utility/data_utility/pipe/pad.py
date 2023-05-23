################################################################################
#
# Utility functions for collating a batch with different number of
# audio frames in each sample.
#
# The functions assume an input dimensionality of [1, NUM_FRAMES] for wave-like input
#
# This will result in batches with respective dimensionality
# [NUM_SAMPLES, MAX_NUM_FRAMES]
#
# Author(s): Nik Vaessen
################################################################################

from typing import List, Callable

import torch as t

from torch.nn import (
    ConstantPad1d,
    ConstantPad2d,
)

################################################################################
# private utility functions


def _determine_max_size(samples: List[t.Tensor], variable_dim: int = 0):
    if len(samples) <= 0:
        raise ValueError("expected non-empty list")

    expected_shape = list(samples[0].shape)
    expected_shape[variable_dim] = None

    max_frames = -1

    for idx, sample in enumerate(samples):
        num_frames = sample.shape[variable_dim]

        assert len(sample.shape) == len(expected_shape)
        assert all(
            [
                sample.shape[dim] == expected_shape[dim]
                for dim in range(len(sample.shape))
                if dim != variable_dim
            ]
        )

        if num_frames > max_frames:
            max_frames = num_frames

    return max_frames


def _generic_append_padding(
    samples: List[t.Tensor],
    padding_init: Callable[[int, int, int], t.nn.Module],
    variable_dim: int = 0,
):
    max_size = _determine_max_size(samples, variable_dim)

    padded_samples = []

    for sample in samples:
        num_dim = len(sample.shape)
        current_size = sample.shape[variable_dim]

        padded_sample = padding_init(num_dim, current_size, max_size)(sample)

        padded_samples.append(padded_sample)

    return t.stack(padded_samples)


################################################################################
# constant collating


def collate_append_constant(
    samples: List[t.Tensor],
    value: float = 0,
    variable_dim: int = 0,
):
    def padding_init(num_dim: int, current_size: int, desired_size: int, v=value):
        padding_right = desired_size - current_size

        if num_dim == 1:
            return ConstantPad1d((0, padding_right), v)
        elif num_dim == 2:
            if variable_dim == 0:
                return ConstantPad2d((0, 0, 0, padding_right), v)
            elif variable_dim == 1:
                return ConstantPad2d((0, padding_right, 0, 0), v)
            else:
                raise ValueError(f"{variable_dim=} can only be 0 or 1")
        else:
            raise ValueError(
                f"only 1 or 2 dimensional tensors are supported, not {num_dim=}"
            )

    return _generic_append_padding(samples, padding_init, variable_dim)
