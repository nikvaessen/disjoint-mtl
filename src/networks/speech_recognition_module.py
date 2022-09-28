################################################################################
#
# Define a base lightning module for a speech recognition network.
#
# Author(s): Nik Vaessen
################################################################################

import json
import logging
import os
import pathlib

from abc import abstractmethod
from typing import Callable, Optional, List, Dict, Tuple, Any

import torch.nn
import torchmetrics

import torch as t

from tqdm import tqdm
from pytorch_lightning.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig

from data_utility.eval.speech.wer import calculate_wer
from data_utility.eval.speech.transform import (
    decode_idx_sequence,
    decode_predictions_greedy,
    decode_idx_sequence_batch,
)
from data_utility.pipe.containers import SpeechRecognitionBatch
from src.networks.base_lightning_module import BaseLightningModule
from src.optim.loss.ctc_loss import CtcLoss

################################################################################
# Definition of speaker recognition API

# A logger for this file

log = logging.getLogger(__name__)


class SpeechRecognitionLightningModule(BaseLightningModule):
    def __init__(
        self,
        hyperparameter_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        idx_to_char: Dict[int, str],
        test_names: List[str],
    ):
        super().__init__(hyperparameter_config, loss_fn_constructor)

        if not isinstance(self.loss_fn, CtcLoss):
            raise ValueError(
                f"expected loss class {CtcLoss}, " f"got {self.loss_fn.__class__}"
            )

        # input arguments
        self.idx_to_char = idx_to_char
        self.test_names = test_names
        self.vocab_size = len(idx_to_char)

        # keep track of metrics
        self.metric_train_loss = torchmetrics.MeanMetric()
        self.metric_train_wer = torchmetrics.MeanMetric()

        self.metric_val_loss = torchmetrics.MeanMetric()

    @abstractmethod
    def compute_embedding_sequence(
        self, input_tensor: t.Tensor, lengths: List[int]
    ) -> Tuple[t.Tensor, List[int]]:
        # transform:
        # 1) input_tensor with shape [BATCH_SIZE, NUM_SAMPLES]
        # 2) where 0:lengths[BATCH_IDX] are non-padded frames
        # into:
        # 1) an embedding of shape [BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_SIZE]
        # 2) a list of lengths which represents frames which are (non-padded)
        #    lengths (index 0:length_value is non-padded)
        pass

    @abstractmethod
    def compute_vocabulary_prediction(
        self, embedding_tensor: t.Tensor, lengths: List[int]
    ) -> Tuple[t.Tensor, List[int]]:
        # transform embedding tensor with shape [BATCH_SIZE, EMBEDDING_SIZE]
        # and list of non-padded range for each batch dimension
        # into a speaker prediction of shape [BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_SIZE]
        # and a list of non-padded range for each batch dimension
        pass

    def forward(self, input_tensor: torch.Tensor, lengths: List[int]):
        embedding, emb_lengths = self.compute_embedding_sequence(input_tensor, lengths)
        prediction, pred_lengths = self.compute_vocabulary_prediction(
            embedding, emb_lengths
        )

        return (embedding, emb_lengths), (prediction, pred_lengths)

    def training_step(
        self,
        batch: SpeechRecognitionBatch,
        batch_idx: int,
        optimized_idx: Optional[int] = None,
    ) -> STEP_OUTPUT:
        assert isinstance(batch, SpeechRecognitionBatch)
        _, (
            letter_prediction,
            letter_prediction_lengths,
        ) = self.forward(batch.audio_tensor, batch.audio_num_frames)

        loss = self.loss_fn(
            predictions=letter_prediction,
            ground_truths=batch.transcriptions_tensor,
            prediction_lengths=letter_prediction_lengths,
            ground_truth_lengths=batch.transcription_length,
        )

        with torch.no_grad():
            predicted_transcriptions = decode_predictions_greedy(
                predictions=letter_prediction,
                until_seq_idx=letter_prediction_lengths,
                idx_to_char=self.idx_to_char,
            )
            label_transcriptions = decode_idx_sequence_batch(
                idx_sequence=batch.transcriptions_tensor,
                until_seq_idx=batch.transcription_length,
                idx_to_char=self.idx_to_char,
            )

            train_wer = calculate_wer(predicted_transcriptions, label_transcriptions)

            # log training loss
            self.metric_train_loss(loss.detach().cpu().item())
            self.metric_train_wer(train_wer)

            if batch_idx == 0:
                with (pathlib.Path.cwd() / "train_predictions.log").open("a") as f:
                    for idx, (pred, gt) in enumerate(
                        zip(predicted_transcriptions, label_transcriptions)
                    ):
                        print(f"{idx:>3d}: {batch.keys[idx]}", file=f)
                        print(f"{idx:>3d}: prediction=`{pred}`", file=f)
                        print(f"{idx:>3d}:      label=`{gt}`", file=f)
                    print(
                        f"{train_wer=}\n",
                        end="\n\n",
                        file=f,
                        flush=True,
                    )

            if batch_idx % 100 == 0:
                self.log_dict(
                    {
                        "train_loss": self.metric_train_loss.compute(),
                        "train_wer": self.metric_train_wer.compute(),
                    },
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                )

                self.metric_train_loss.reset()
                self.metric_train_wer.reset()

        return loss

    def validation_step(
        self,
        batch: SpeechRecognitionBatch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Optional[STEP_OUTPUT]:
        assert isinstance(batch, SpeechRecognitionBatch)

        _, (
            letter_prediction,
            letter_prediction_lengths,
        ) = self.forward(batch.audio_tensor, batch.audio_num_frames)

        loss = self.loss_fn(
            predictions=letter_prediction,
            ground_truths=batch.transcriptions_tensor,
            prediction_lengths=letter_prediction_lengths,
            ground_truth_lengths=batch.transcription_length,
        )

        self.metric_val_loss(loss.detach().cpu().item())

        with torch.no_grad():
            predicted_transcriptions = decode_predictions_greedy(
                predictions=letter_prediction,
                until_seq_idx=letter_prediction_lengths,
                idx_to_char=self.idx_to_char,
            )
            label_transcriptions = decode_idx_sequence_batch(
                idx_sequence=batch.transcriptions_tensor,
                until_seq_idx=batch.transcription_length,
                idx_to_char=self.idx_to_char,
            )

        return {
            "transcription": predicted_transcriptions,
            "ground_truth": label_transcriptions,
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        val_wer = self.calculate_wer_on_collected_output(outputs)

        self.log_dict(
            {"val_loss": self.metric_val_loss.compute(), "val_wer": val_wer},
            on_epoch=True,
            prog_bar=True,
        )
        self.metric_val_loss.reset()

    def test_step(
        self,
        batch: SpeechRecognitionBatch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Optional[STEP_OUTPUT]:
        assert isinstance(batch, SpeechRecognitionBatch)

        (embedding, embedding_lengths), (
            letter_prediction,
            letter_prediction_lengths,
        ) = self.forward(batch.audio_tensor, batch.audio_num_frames)

        with torch.no_grad():
            predicted_transcriptions = decode_predictions_greedy(
                predictions=letter_prediction,
                until_seq_idx=letter_prediction_lengths,
                idx_to_char=self.idx_to_char,
            )
            label_transcriptions = decode_idx_sequence_batch(
                idx_sequence=batch.transcriptions_tensor,
                until_seq_idx=batch.transcription_length,
                idx_to_char=self.idx_to_char,
            )

        return {
            "transcription": predicted_transcriptions,
            "ground_truth": label_transcriptions,
        }

    def test_epoch_end(self, outputs: List[Dict]) -> None:
        if len(self.test_names) == 1:
            outputs = [outputs]

        result_dict = {}

        for idx in range(len(outputs)):
            key = self.test_names[idx]

            wer = self.calculate_wer_on_collected_output(outputs[idx])

            result_dict[f"test_wer_{key}"] = wer

        self.log_dict(result_dict)

    @staticmethod
    def calculate_wer_on_collected_output(list_of_dict: List[Dict]):
        all_transcriptions = []
        all_ground_truths = []

        for d in list_of_dict:
            all_transcriptions.extend(d["transcription"])
            all_ground_truths.extend(d["ground_truth"])

        wer = calculate_wer(
            transcriptions=all_transcriptions, ground_truths=all_ground_truths
        )

        return wer
