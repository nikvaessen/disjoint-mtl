########################################################################################
#
# Define a base lightning module for a MTL speech and speaker recognition network
# with disjoint training.
#
# Author(s): Nik Vaessen
########################################################################################

import logging
import pathlib

from abc import abstractmethod
from collections import defaultdict
from typing import Callable, Optional, List, Dict, Tuple, Any, Union, Iterator

import math
import torch.nn
import torchmetrics

import torch as t

from pytorch_lightning.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from scipy.optimize import minimize_scalar
from torch.nn import Parameter

from data_utility.eval.speaker.cosine_dist_evaluator import CosineDistanceEvaluator
from data_utility.eval.speaker.evaluator import SpeakerTrial
from data_utility.eval.speech.wer import calculate_wer
from data_utility.eval.speech.transform import (
    decode_predictions_greedy,
)
from data_utility.pipe.containers import (
    SpeechRecognitionBatch,
    SpeakerRecognitionBatch,
)
from src.network.base_lightning_module import BaseLightningModule
from src.network.speaker_recognition_module import evaluate_embeddings
from src.optim.loss.mt_speech_speaker_loss import MTSpeechAndSpeakerLoss

########################################################################################
# Definition of speaker recognition API

# A logger for this file

log = logging.getLogger(__name__)


def avg_grad(g1: t.Tensor, g2: t.Tensor):
    with torch.no_grad():
        return (g1 + g2) / 2


def ca_grad(g1: t.Tensor, g2: t.Tensor, c: t.Tensor):
    with torch.no_grad():
        g0 = (g1 + g2) / 2
        g0_norm = torch.linalg.norm(g0)
        phi = c**2 * g0_norm**2
        phi_sqrt = torch.sqrt(phi)

        def min_fn(w):
            w1 = w
            w2 = 1 - w

            gw_temp = (w1 * g1) + (w2 * g2) / 2
            gw_norm_temp = torch.linalg.norm(gw_temp)

            gwg0 = torch.dot(gw_temp, g0)

            objective = gwg0 + (phi_sqrt * gw_norm_temp)

            return objective.cpu().detach().item()

        res = minimize_scalar(min_fn, bounds=(0, 1), method="bounded")
        opt_w1 = res.x
        opt_w2 = 1 - opt_w1

        gw = (opt_w1 * g1) + (opt_w2 * g2)
        gw_norm = torch.linalg.norm(gw)
        coef = phi_sqrt / gw_norm
        g = g0 + (coef * gw)

        return g


class DisjointMTLLightningModule(BaseLightningModule):
    def __init__(
        self,
        hyperparameter_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        idx_to_char: Dict[int, str],
        val_names: List[str],
        val_modes: List[str],
        test_names: List[str],
        test_modes: List[str],
        num_speakers: int,
        test_pairs: List[List[SpeakerTrial]],
        apply_ca_grad: bool,
        ca_grad_c: float,
    ):
        super().__init__(hyperparameter_config, loss_fn_constructor)

        if not isinstance(self.loss_fn, MTSpeechAndSpeakerLoss):
            raise ValueError(
                f"expected loss class {MTSpeechAndSpeakerLoss}, got {self.loss_fn.__class__}"
            )

        # input arguments
        self.idx_to_char = idx_to_char
        self.vocab_size = len(idx_to_char)
        self.num_speakers = num_speakers

        self.val_names = val_names
        self.val_modes = val_modes
        assert len(val_names) == len(val_modes) == 2

        self.test_names = test_names
        self.test_modes = test_modes
        self.test_pairs = test_pairs
        assert len(test_names) == len(test_modes) == len(test_pairs)

        # opt settings
        self.apply_ca_grad = apply_ca_grad
        self.ca_grad_c = t.tensor(ca_grad_c)

        # evaluator
        self.evaluator = CosineDistanceEvaluator(
            center_before_scoring=False,
            length_norm_before_scoring=False,
        )

        # keep track of metrics
        self.metric_train_loss = torchmetrics.MeanMetric()
        self.metric_train_speech_loss = torchmetrics.MeanMetric()
        self.metric_train_speaker_loss = torchmetrics.MeanMetric()
        self.train_grad_dict = defaultdict(list)

        self.metric_train_acc = torchmetrics.Accuracy()
        self.metric_train_wer = torchmetrics.MeanMetric()

        self.metric_val_loss = torchmetrics.MeanMetric()
        self.metric_val_speech_loss = torchmetrics.MeanMetric()
        self.metric_val_speaker_loss = torchmetrics.MeanMetric()

        self.metric_val_acc = torchmetrics.Accuracy()

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

    @abstractmethod
    def shared_params(self) -> Iterator[Tuple[str, Parameter]]:
        return self.named_parameters()

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

    def grad2vec(self, apply_zero_grad: bool = True):
        with torch.no_grad():
            # extract all gradients from shared parameters and put them into a single vector
            reconstruction_dict = {}
            stack = []
            start_idx = 0

            for name, param in self.shared_params():
                if param.requires_grad and param.grad is not None:
                    reconstruction_dict[name] = start_idx
                    flat_grad = param.grad.flatten()
                    stack.append(flat_grad)
                    start_idx += flat_grad.shape[0]

                    if apply_zero_grad:
                        param.grad = None

            if len(stack) == 0:  # network is frozen
                return None, None
            else:
                grad_vec = torch.concat(stack)

                return grad_vec, reconstruction_dict

    def vec2grad(self, vec: torch.Tensor, reconstruction_dict: Dict[str, int]):
        with torch.no_grad():
            # put the single grad vector back into the grad of all shared parameters
            for name, param in self.shared_params():
                if name in reconstruction_dict:
                    num_params = math.prod(param.shape)
                    begin_idx = reconstruction_dict[name]
                    end_idx = begin_idx + num_params

                    flattened_vec = vec[begin_idx:end_idx]
                    param.grad = flattened_vec.unflatten(-1, param.shape)

    def training_step(
        self,
        batch: Tuple[SpeechRecognitionBatch, SpeakerRecognitionBatch],
        batch_idx: int,
        optimized_idx: Optional[int] = None,
    ):
        asr_batch, sv_batch = batch
        assert isinstance(asr_batch, SpeechRecognitionBatch)
        assert isinstance(sv_batch, SpeakerRecognitionBatch)

        # forward step for task 1 (asr)
        (_, (asr_prediction, asr_pred_lengths), _) = self.forward(
            asr_batch.audio_tensor, asr_batch.audio_num_frames
        )
        loss_speech = self.loss_fn.speech_loss(
            predictions=asr_prediction,
            ground_truths=asr_batch.transcriptions_tensor,
            prediction_lengths=asr_pred_lengths,
            ground_truth_lengths=asr_batch.transcriptions_length,
        )

        # forward step for task 2 (sv)
        (_, _, (sv_embedding, sv_logits)) = self.forward(
            sv_batch.audio_tensor, sv_batch.audio_num_frames
        )
        loss_speaker, sv_prediction = self.loss_fn.speaker_loss(
            sv_logits, sv_batch.id_tensor
        )

        # combine gradients for task 1 and 2
        self.manual_backward(loss_speech)
        g1, g1_dict = self.grad2vec()
        self.manual_backward(loss_speaker)
        g2, g2_dict = self.grad2vec()

        if g1 is not None and g2 is not None:
            # network is not frozen
            if self.apply_ca_grad:
                g0 = ca_grad(g1, g2, self.ca_grad_c)
            else:
                g0 = avg_grad(g1, g2)

            self.vec2grad(g0, g1_dict)
        else:
            g0 = None

        print(f"\n### {batch_idx=} ###\n{g0=}\n{g1=}\n{g2=}\n\n")

        # step weights
        opt = self.optimizers()
        lr_schedule = self.lr_schedulers()

        opt.step()
        lr_schedule.step()
        self.zero_grad()

        # log values
        with torch.no_grad():
            predicted_transcriptions = decode_predictions_greedy(
                predictions=asr_prediction,
                until_seq_idx=asr_pred_lengths,
                idx_to_char=self.idx_to_char,
            )
            label_transcriptions = asr_batch.transcriptions
            train_wer = calculate_wer(predicted_transcriptions, label_transcriptions)

            self._log_train_acc(sv_prediction, sv_batch.id_tensor, batch_idx)
            self._log_train_predictions(
                asr_batch,
                batch_idx,
                predicted_transcriptions,
                label_transcriptions,
                train_wer,
            )
            self._log_train_wer(train_wer, batch_idx)
            self._log_train_loss(
                (loss_speaker + loss_speech) / 2, loss_speaker, loss_speech, batch_idx
            )
            self._log_train_gradients(batch_idx, g0, g1, g2)

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

    def _log_train_gradients(
        self, batch_idx: int, g0: t.Tensor, g1: t.Tensor, g2: t.Tensor
    ):
        if g0 is None or g1 is None or g2 is None:
            return

        g0g1 = torch.dot(g0, g1)
        g0g2 = torch.dot(g0, g2)
        g1g1 = torch.dot(g1, g2)

        self.train_grad_dict["g0g1"].append(g0g1)
        self.train_grad_dict["g0g2"].append(g0g2)
        self.train_grad_dict["g1g2"].append(g1g1)

        if batch_idx % 100 == 0:
            for key in ["g0g1", "g0g2", "g1g2"]:
                value_list = self.train_grad_dict[key]
                self.train_grad_dict[key] = []

                value_tensor = t.tensor(value_list)

                self.log(f"{key}_min", t.min(value_tensor))
                self.log(f"{key}_max", t.max(value_tensor))
                self.log(f"{key}_avg", t.mean(value_tensor))
                self.log(f"{key}_pos_count", t.sum(value_tensor > 0))

    def validation_step(
        self,
        batch: Union[SpeechRecognitionBatch, SpeakerRecognitionBatch],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Optional[STEP_OUTPUT]:
        mode = self.val_modes[dataloader_idx]

        (
            _,
            (asr_prediction, asr_pred_lengths),
            (speaker_embedding, speaker_prediction),
        ) = self.forward(batch.audio_tensor, batch.audio_num_frames)

        if mode == "speaker":
            assert isinstance(batch, SpeakerRecognitionBatch)

            loss_speaker, sv_prediction = self.loss_fn.speaker_loss(
                speaker_prediction, batch.id_tensor
            )

            self.metric_val_acc(sv_prediction, batch.id_tensor)
            self.metric_val_speaker_loss(loss_speaker)

            return {}

        elif mode == "speech":
            assert isinstance(batch, SpeechRecognitionBatch)

            loss_speech = self.loss_fn.speech_loss(
                predictions=asr_prediction,
                ground_truths=batch.transcriptions_tensor,
                prediction_lengths=asr_pred_lengths,
                ground_truth_lengths=batch.transcriptions_length,
            )

            self.metric_val_speech_loss(loss_speech)

            predicted_transcriptions = decode_predictions_greedy(
                predictions=asr_prediction,
                until_seq_idx=asr_pred_lengths,
                idx_to_char=self.idx_to_char,
            )

            return {
                "transcription": predicted_transcriptions,
                "ground_truth": batch.transcriptions,
            }

        else:
            raise ValueError(f"unknown validation mode `{mode}`")

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        wer_idx = 0 if self.val_modes[0] == "speech" else 1
        val_wer = self.calculate_wer_on_collected_output(outputs[wer_idx])

        val_speaker_loss = self.metric_val_speaker_loss.compute()
        val_speech_loss = self.metric_val_speech_loss.compute()
        val_loss = val_speech_loss + val_speaker_loss

        self.log_dict(
            {
                "val_loss": val_loss,
                "val_speaker_loss": val_speaker_loss,
                "val_speech_loss": val_speech_loss,
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

    def test_step(
        self,
        batch: Union[SpeechRecognitionBatch, SpeakerRecognitionBatch],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Optional[STEP_OUTPUT]:
        mode = self.test_modes[dataloader_idx]

        (
            _,
            (asr_prediction, asr_pred_lengths),
            (speaker_embedding, speaker_prediction),
        ) = self.forward(batch.audio_tensor, batch.audio_num_frames)

        if mode == "speaker":
            assert isinstance(batch, SpeakerRecognitionBatch)

            return {
                "embedding": speaker_embedding.detach().cpu(),
                "sample_id": batch.keys,
            }

        elif mode == "speech":
            assert isinstance(batch, SpeechRecognitionBatch)

            predicted_transcriptions = decode_predictions_greedy(
                predictions=asr_prediction,
                until_seq_idx=asr_pred_lengths,
                idx_to_char=self.idx_to_char,
            )

            return {
                "transcription": predicted_transcriptions,
                "ground_truth": batch.transcriptions,
            }

        else:
            raise ValueError(f"unknown validation mode `{mode}`")

    def test_epoch_end(self, outputs: List[Dict]) -> None:
        if len(self.test_names) == 1:
            outputs = [outputs]

        result_dict = {}

        for idx in range(len(outputs)):
            key = self.test_names[idx]
            mode = self.test_modes[idx]

            if mode == "speaker":
                results = evaluate_embeddings(
                    self.evaluator, outputs[idx], self.test_pairs[idx], True
                )
                result_dict[f"test_eer_{key}"] = results["eer"]
                result_dict[f"test_eer_threshold_{key}"] = results["eer_threshold"]

            elif mode == "speech":
                wer = self.calculate_wer_on_collected_output(outputs[idx])
                result_dict[f"test_wer_{key}"] = wer

            else:
                raise ValueError(f"unknown test mode {mode}")

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
