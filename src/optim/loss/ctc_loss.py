################################################################################
#
# CTC loss for speech recognition
#
# Author(s): Nik Vaessen
################################################################################

from typing import List

import torch as t

import torch.nn as nn
import torch.nn.functional as F


################################################################################
# wrapper around ctc loss of pytorch


class CtcLoss(nn.Module):
    def __init__(self, ctc_blank_idx: int = 0):
        super().__init__()

        self.blank_idx = ctc_blank_idx

    def forward(
            self,
            predictions: t.Tensor,
            prediction_lengths: List[int],
            ground_truths: t.Tensor,
            ground_truth_lengths: List[int],
    ):
        original_device = predictions.device
        assert original_device == predictions.device == ground_truths.device

        # predictions will be shape [BATCH_SIZE, MAX_INPUT_SEQUENCE_LENGTH, CLASSES]
        # expected to be [MAX_INPUT_SEQUENCE_LENGTH, BATCH_SIZE, CLASSES] for
        # loss function
        predictions = t.transpose(predictions, 0, 1)

        # they also need to be log probabilities
        predictions = F.log_softmax(predictions, dim=2)

        # prediction lengths will be shape [BATCH_SIZE]
        prediction_lengths = t.tensor(prediction_lengths, dtype=t.long)
        assert len(prediction_lengths.shape) == 1
        assert prediction_lengths.shape[0] == predictions.shape[1]

        # ground truths will be shape [BATCH_SIZE, MAX_TARGET_SEQUENCE_LENGTH]
        assert len(ground_truths.shape) == 2
        assert ground_truths.shape[0] == predictions.shape[1]

        # ground_truth_lengths will be shape [BATCH_SIZE]
        ground_truth_lengths = t.tensor(ground_truth_lengths, dtype=t.long)
        assert len(ground_truth_lengths.shape) == 1
        assert ground_truth_lengths.shape[0] == predictions.shape[1]

        # ctc loss expects every tensor to be on CPU
        # we disable cudnn due to variable input lengths
        with t.backends.cudnn.flags(enabled=False):
            return F.ctc_loss(
                log_probs=predictions.to(
                    original_device,
                ),
                targets=ground_truths.to(original_device),
                input_lengths=prediction_lengths.to(original_device),
                target_lengths=ground_truth_lengths.to(original_device),
                blank=self.blank_idx,
                zero_infinity=True,  # prevents any weird crashes
            ).to(original_device)
