########################################################################################
#
# Define a base lightning module for a MTL speech and speaker recognition network.
#
# Author(s): Nik Vaessen
########################################################################################

import logging
import pathlib

from abc import abstractmethod
from typing import Callable, Optional, List, Dict, Tuple, Any

import torch.nn
import torchmetrics

import torch as t

from pytorch_lightning.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig

from data_utility.eval.speaker.cosine_dist_evaluator import CosineDistanceEvaluator
from data_utility.eval.speaker.evaluator import SpeakerTrial
from data_utility.eval.speech.wer import calculate_wer
from data_utility.eval.speech.transform import (
    decode_predictions_greedy,
)
from data_utility.pipe.containers import (
    SpeechRecognitionBatch,
    SpeechAndSpeakerRecognitionBatch,
)
from src.networks.base_lightning_module import BaseLightningModule
from src.networks.speaker_recognition_module import evaluate_embeddings
from src.optim.loss.mt_speech_speaker_loss import MTSpeechAndSpeakerLoss

########################################################################################
# Definition of speaker recognition API

# A logger for this file

log = logging.getLogger(__name__)


class MTLLightningModule(BaseLightningModule):
    def __init__(
        self,
        hyperparameter_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        idx_to_char: Dict[int, str],
        test_names: List[str],
        num_speakers: int,
        test_pairs: List[List[SpeakerTrial]],
    ):
        super().__init__(hyperparameter_config, loss_fn_constructor)

        if not isinstance(self.loss_fn, MTSpeechAndSpeakerLoss):
            raise ValueError(
                f"expected loss class {MTSpeechAndSpeakerLoss}, got {self.loss_fn.__class__}"
            )

        # input arguments
        self.idx_to_char = idx_to_char
        self.test_names = test_names
        self.vocab_size = len(idx_to_char)
        self.num_speakers = num_speakers
        self.test_pairs = test_pairs

        # evaluator
        self.evaluator = CosineDistanceEvaluator(
            center_before_scoring=False,
            length_norm_before_scoring=False,
        )

        # keep track of metrics
        self.metric_train_loss = torchmetrics.MeanMetric()
        self.metric_train_speech_loss = torchmetrics.MeanMetric()
        self.metric_train_speaker_loss = torchmetrics.MeanMetric()

        self.metric_train_acc = torchmetrics.Accuracy()
        self.metric_train_wer = torchmetrics.MeanMetric()

        self.metric_val_loss = torchmetrics.MeanMetric()
        self.metric_val_speech_loss = torchmetrics.MeanMetric()
        self.metric_val_speaker_loss = torchmetrics.MeanMetric()

        self.metric_val_acc = torchmetrics.Accuracy()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

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
        self, sequence_tensor: t.Tensor, lengths: List[int]
    ) -> Tuple[t.Tensor, List[int]]:
        # transform embedding tensor with shape [BATCH_SIZE, EMBEDDING_SIZE]
        # and list of non-padded range for each batch dimension
        # into a speaker prediction of shape [BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_SIZE]
        # and a list of non-padded range for each batch dimension
        pass

    @property
    @abstractmethod
    def speaker_embedding_size(self):
        pass

    @abstractmethod
    def compute_speaker_prediction(self, embedding_tensor: t.Tensor) -> t.Tensor:
        # transform embedding tensor with shape [BATCH_SIZE, EMBEDDING_SIZE]
        # into a speaker prediction of shape [BATCH_SIZE, NUM_SPEAKERS]
        pass

    @abstractmethod
    def compute_speaker_embedding(self, sequence_tensor: t.Tensor) -> t.Tensor:
        # transform input_tensor with shape [BATCH_SIZE, ...]
        # into an embedding of shape [BATCH_SIZE, EMBEDDING_SIZE]
        pass

    def forward(self, input_tensor: torch.Tensor, lengths: List[int]):
        sequence, sequence_lengths = self.compute_embedding_sequence(
            input_tensor, lengths
        )

        asr_prediction, asr_pred_lengths = self.compute_vocabulary_prediction(
            sequence, sequence_lengths
        )

        speaker_embedding = self.compute_speaker_embedding(sequence)
        speaker_prediction = self.compute_speaker_prediction(speaker_embedding)

        return (
            (sequence, sequence_lengths),
            (asr_prediction, asr_pred_lengths),
            (speaker_embedding, speaker_prediction),
        )

    def training_step(
        self,
        batch: SpeechAndSpeakerRecognitionBatch,
        batch_idx: int,
        optimized_idx: Optional[int] = None,
    ) -> STEP_OUTPUT:
        assert isinstance(batch, SpeechAndSpeakerRecognitionBatch)
        (
            _,
            (asr_prediction, asr_pred_lengths),
            (speaker_embedding, speaker_logits),
        ) = self.forward(batch.audio_tensor, batch.audio_num_frames)

        (
            summed_loss,
            speech_loss_value,
            speaker_loss_value,
            speaker_prediction,
            speech_weight,
            speaker_weight,
        ) = self.loss_fn(
            speech_predictions=asr_prediction,
            speech_prediction_lengths=asr_pred_lengths,
            speech_ground_truths=batch.transcriptions_tensor,
            speech_ground_truth_lengths=batch.transcriptions_length,
            speaker_logits=speaker_logits,
            speaker_labels=batch.id_tensor,
        )

        # make logs
        with torch.no_grad():
            predicted_transcriptions = decode_predictions_greedy(
                predictions=asr_prediction,
                until_seq_idx=asr_pred_lengths,
                idx_to_char=self.idx_to_char,
            )
            label_transcriptions = batch.transcriptions

            train_wer = calculate_wer(predicted_transcriptions, label_transcriptions)

            # speech
            self._log_train_predictions(
                batch,
                batch_idx,
                predicted_transcriptions,
                label_transcriptions,
                train_wer,
            )
            self._log_train_wer(train_wer, batch_idx)

            # speaker
            self._log_train_acc(speaker_prediction, batch.id_tensor, batch_idx)
            self._log_train_batch_info(batch)

            # loss
            self._log_train_loss(
                summed_loss, speaker_loss_value, speech_loss_value, batch_idx
            )

        # manual optimization
        opt = self.optimizers()
        lr_scheduler = self.lr_schedulers()

        opt.zero_grad()
        self.manual_backward(summed_loss)

        opt.step()
        lr_scheduler.step()

    def _log_train_batch_info(self, batch):
        with (pathlib.Path.cwd() / "train_batch_info.log").open("a") as f:
            print(
                f"{batch.batch_size=} "
                f"{batch.audio_tensor.shape=} "
                f"{batch.id_tensor.shape=}",
                file=f,
            )

    def _log_train_acc(self, prediction: t.Tensor, label: t.Tensor, batch_idx: int):
        self.metric_train_acc(prediction, label)

        if batch_idx % 100 == 0:
            self.log(
                "train_acc",
                self.metric_train_acc.compute(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            self.metric_train_acc.reset()

    def _log_train_predictions(
        self,
        batch,
        batch_idx,
        predicted_transcriptions,
        label_transcriptions,
        train_wer,
    ):
        if batch_idx % 5000 == 0:
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

    def _log_train_wer(self, train_wer: float, batch_idx: int):
        self.metric_train_wer(train_wer)

        if batch_idx % 100 == 0:
            self.log(
                "train_wer",
                self.metric_train_wer.compute(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            self.metric_train_wer.reset()

    def _log_train_loss(
        self,
        loss: t.Tensor,
        speaker_loss: t.Tensor,
        speech_loss: t.Tensor,
        batch_idx: int,
    ):
        self.metric_train_loss(loss)
        self.metric_train_speaker_loss(speaker_loss)
        self.metric_train_speech_loss(speech_loss)

        if batch_idx % 100 == 0:
            self.log_dict(
                {
                    "train_loss": self.metric_train_loss.compute(),
                    "train_speaker_loss": self.metric_train_speaker_loss.compute(),
                    "train_speech_loss": self.metric_train_speech_loss.compute(),
                },
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )

            self.metric_train_loss.reset()
            self.metric_train_speaker_loss.reset()
            self.metric_train_speech_loss.reset()

    def validation_step(
        self,
        batch: SpeechAndSpeakerRecognitionBatch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Optional[STEP_OUTPUT]:
        assert isinstance(batch, SpeechAndSpeakerRecognitionBatch)

        (
            _,
            (asr_prediction, asr_pred_lengths),
            (speaker_embedding, speaker_logits),
        ) = self.forward(batch.audio_tensor, batch.audio_num_frames)

        (
            summed_loss,
            speech_loss_value,
            speaker_loss_value,
            speaker_prediction,
            speech_weight,
            speaker_weight,
        ) = self.loss_fn(
            speech_predictions=asr_prediction,
            speech_prediction_lengths=asr_pred_lengths,
            speech_ground_truths=batch.transcriptions_tensor,
            speech_ground_truth_lengths=batch.transcriptions_length,
            speaker_logits=speaker_logits,
            speaker_labels=batch.id_tensor,
        )

        with torch.no_grad():
            predicted_transcriptions = decode_predictions_greedy(
                predictions=asr_prediction,
                until_seq_idx=asr_pred_lengths,
                idx_to_char=self.idx_to_char,
            )
            label_transcriptions = batch.transcriptions

            val_wer = calculate_wer(predicted_transcriptions, label_transcriptions)

            # speech
            self._log_val_predictions(
                batch,
                batch_idx,
                predicted_transcriptions,
                label_transcriptions,
                val_wer,
            )
            self._log_val_batch_info(batch)

            # values
            self.metric_val_acc(speaker_prediction, batch.id_tensor)
            self.metric_val_loss(summed_loss)
            self.metric_val_speaker_loss(speaker_loss_value)
            self.metric_val_speech_loss(speech_loss_value)

        return {
            "transcription": predicted_transcriptions,
            "ground_truth": label_transcriptions,
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        val_wer = self.calculate_wer_on_collected_output(outputs)

        self.log_dict(
            {
                "val_loss": self.metric_val_loss.compute(),
                "val_speaker_loss": self.metric_val_speaker_loss.compute(),
                "val_speech_loss": self.metric_val_speech_loss.compute(),
                "val_wer": val_wer,
                "val_acc": self.metric_val_acc.compute(),
            },
            on_epoch=True,
            prog_bar=True,
        )

        self.metric_val_loss.reset()
        self.metric_val_speaker_loss.reset()
        self.metric_val_speech_loss.reset()
        self.metric_val_acc.reset()

    def _log_val_batch_info(self, batch):
        with (pathlib.Path.cwd() / "val_batch_info.log").open("a") as f:
            print(
                f"{batch.batch_size=} "
                f"{batch.audio_tensor.shape=} "
                f"{batch.id_tensor.shape=}",
                file=f,
            )

    def _log_val_predictions(
        self,
        batch,
        batch_idx,
        predicted_transcriptions,
        label_transcriptions,
        train_wer,
    ):
        if batch_idx == 0:
            with (pathlib.Path.cwd() / "val_predictions.log").open("a") as f:
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

    def test_step(
        self,
        batch: SpeechAndSpeakerRecognitionBatch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Optional[STEP_OUTPUT]:
        assert isinstance(batch, SpeechAndSpeakerRecognitionBatch)

        (
            _,
            (asr_prediction, asr_pred_lengths),
            (speaker_embedding, speaker_logits),
        ) = self.forward(batch.audio_tensor, batch.audio_num_frames)

        with torch.no_grad():
            predicted_transcriptions = decode_predictions_greedy(
                predictions=asr_prediction,
                until_seq_idx=asr_pred_lengths,
                idx_to_char=self.idx_to_char,
            )
            label_transcriptions = batch.transcriptions

        if (
            len(speaker_embedding.shape) == 1
            and speaker_embedding.shape[0] == self.speaker_embedding_size
        ):
            speaker_embedding = speaker_embedding[None, :]

        assert len(speaker_embedding.shape) == 2
        assert speaker_embedding.shape[0] == batch.batch_size
        assert speaker_embedding.shape[1] == self.speaker_embedding_size

        speaker_embedding = t.stack([speaker_embedding.detach().to("cpu")])

        return {
            "transcription": predicted_transcriptions,
            "ground_truth": label_transcriptions,
            "embedding": speaker_embedding,
            "sample_id": batch.keys,
        }

    def test_epoch_end(self, outputs: List[Dict]) -> None:
        if len(self.test_names) == 1:
            outputs = [outputs]

        result_dict = {}

        for idx in range(len(outputs)):
            key = self.test_names[idx]

            wer = self.calculate_wer_on_collected_output(outputs[idx])
            results = evaluate_embeddings(
                self.evaluator, outputs[idx], self.test_pairs[idx], True
            )

            result_dict[f"test_wer_{key}"] = wer
            result_dict[f"test_eer_{key}"] = results["eer"]
            result_dict[f"test_eer_threshold_{key}"] = results["eer_threshold"]

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
