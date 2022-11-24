################################################################################
#
# Implement a multi-task learning loss with CTC loss for speech and AAM
# softmax loss for speaker recognition
#
# Author(s): Nik Vaessen
################################################################################

import torch as t

from typing import List, Optional
from torch import nn

from src.optim.loss import CtcLoss, AngularAdditiveMarginSoftMaxLoss, CrossEntropyLoss
from src.optim.loss.mt_weighting import (
    DynamicScaling,
    DynamicWeightAveraging,
    StaticScaling,
)

################################################################################
# implementation


class MTSpeechAndSpeakerLoss(nn.Module):
    def __init__(
        self,
        use_cross_entropy: bool = True,  # ignore aam_scale and aam_margin values if true
        aam_margin: Optional[float] = 0.3,
        aam_scale: Optional[float] = 15,
        ctc_blank_idx: int = 0,
        scale_method: Optional[str] = None,  # one of 'min', 'max', 'static', 'dwa'
        static_speech_weight: Optional[
            float
        ] = 0.5,  # and 1-static_speech_weight for static_speaker_weight
        dwa_temperature: Optional[float] = 1,
        dwa_weight_sum: Optional[float] = 1,
        dwa_use_average_loss: Optional[bool] = True,
        dwa_average_loss_window: Optional[int] = 10,
    ):

        super().__init__()

        # setup dynamic scaling
        if scale_method in ["min", "max"]:
            self.scaler = DynamicScaling(mode=scale_method)
        elif scale_method == "dwa":
            self.scaler = DynamicWeightAveraging(
                num_losses=2,
                use_average_loss=dwa_use_average_loss,
                average_loss_window=dwa_average_loss_window,
                weight_sum=dwa_weight_sum,
                temperature=dwa_temperature,
            )
        elif scale_method == "static":
            self.scaler = StaticScaling(
                weights=[static_speech_weight, 1 - static_speech_weight]
            )
        elif scale_method is None:
            self.scaler = None
        else:
            raise ValueError(
                f"{scale_method=} should be one of 'min','max, 'dwa', 'static', or be None"
            )

        # set up losses
        self.speech_loss = CtcLoss(ctc_blank_idx=ctc_blank_idx)

        if use_cross_entropy:
            self.speaker_loss = CrossEntropyLoss()
        else:
            self.speaker_loss = AngularAdditiveMarginSoftMaxLoss(
                margin=aam_margin,
                scale=aam_scale,
            )

    def compute_speech_loss(
        self,
        speech_predictions: t.Tensor,
        speech_prediction_lengths: List[int],
        speech_ground_truths: t.Tensor,
        speech_ground_truth_lengths: List[int],
    ):
        return self.speech_loss(
            predictions=speech_predictions,
            ground_truths=speech_ground_truths,
            prediction_lengths=speech_prediction_lengths,
            ground_truth_lengths=speech_ground_truth_lengths,
        )

    def compute_speaker_loss(
        self,
        speaker_logits: t.Tensor,
        speaker_labels: t.Tensor,
    ):
        speaker_loss_value, speaker_prediction = self.speaker_loss(
            speaker_logits, speaker_labels
        )

        return speaker_loss_value, speaker_prediction

    def compute_scale(self, speech_loss_value, speaker_loss_value):
        if self.scaler is not None:
            (
                speech_weight,
                speaker_weight,
            ) = self.scaler(speech_loss_value, speaker_loss_value)
        else:
            speech_weight = t.tensor(1)
            speaker_weight = t.tensor(1)

        return speech_weight, speaker_weight

    def forward(
        self,
        speech_predictions: t.Tensor,
        speech_prediction_lengths: List[int],
        speech_ground_truths: t.Tensor,
        speech_ground_truth_lengths: List[int],
        speaker_logits: t.Tensor,
        speaker_labels: t.Tensor,
        apply_scaling: bool = False,
    ):
        speech_loss_value = self.compute_speech_loss(
            speech_predictions,
            speech_prediction_lengths,
            speech_ground_truths,
            speech_ground_truth_lengths,
        )

        speaker_loss_value, speaker_prediction = self.compute_speaker_loss(
            speaker_logits, speaker_labels
        )

        speech_weight, speaker_weight = self.compute_scale(
            speech_loss_value, speaker_loss_value
        )

        if apply_scaling:
            speech_loss_value = speech_loss_value * speech_weight
            speaker_loss_value = speaker_loss_value * speaker_weight

        return (
            speech_loss_value,
            speaker_loss_value,
            speaker_prediction,
            speech_weight,
            speaker_weight,
        )
