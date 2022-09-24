################################################################################
#
# Define a base lightning module for a speaker recognition network.
#
# Author(s): Nik Vaessen
################################################################################

import json
import logging
import pathlib

from abc import abstractmethod
from typing import Callable, Optional, List, Any, Dict

import torch as t
import torchmetrics

from omegaconf import DictConfig

from data_utility.eval.speaker.cosine_dist_evaluator import CosineDistanceEvaluator
from data_utility.eval.speaker.evaluator import (
    SpeakerRecognitionEvaluator,
    EmbeddingSample,
    SpeakerTrial,
)
from data_utility.pipe.containers import SpeakerRecognitionBatch
from src.networks.base_lightning_module import BaseLightningModule

################################################################################
# Definition of speaker recognition API

# A logger for this file

log = logging.getLogger(__name__)


class SpeakerRecognitionLightningModule(BaseLightningModule):
    def __init__(
        self,
        root_hydra_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        num_speakers: int,
        test_pairs: List[List[SpeakerTrial]],
        test_names: List[str],
    ):
        super().__init__(root_hydra_config, loss_fn_constructor)

        # input arguments
        self.num_speakers = num_speakers
        self.test_pairs = test_pairs
        self.test_names = test_names

        # used to keep track of training/val accuracy
        self.metric_train_acc = torchmetrics.Accuracy()
        self.metric_train_loss = torchmetrics.MeanMetric()
        self.metric_valid_acc = torchmetrics.Accuracy()

        # evaluator
        self.evaluator = CosineDistanceEvaluator(
            center_before_scoring=False,
            length_norm_before_scoring=False,
        )

    @property
    @abstractmethod
    def speaker_embedding_size(self):
        pass

    @abstractmethod
    def compute_speaker_embedding(self, input_tensor: t.Tensor) -> t.Tensor:
        # transform input_tensor with shape [BATCH_SIZE, ...]
        # into an embedding of shape [BATCH_SIZE, EMBEDDING_SIZE]
        pass

    @abstractmethod
    def compute_speaker_prediction(self, embedding_tensor: t.Tensor) -> t.Tensor:
        # transform embedding tensor with shape [BATCH_SIZE, EMBEDDING_SIZE]
        # into a speaker prediction of shape [BATCH_SIZE, NUM_SPEAKERS]
        pass

    def forward(self, input_tensor: t.Tensor):
        embedding = self.compute_speaker_embedding(input_tensor)
        prediction = self.compute_speaker_prediction(embedding)

        return embedding, prediction

    def training_step(
        self,
        batch: SpeakerRecognitionBatch,
        batch_idx: int,
        optimized_idx: Optional[int] = None,
    ):
        assert isinstance(batch, SpeakerRecognitionBatch)

        audio_input = batch.audio_tensor
        spk_label = batch.id_tensor

        embedding = self.compute_speaker_embedding(audio_input)

        assert len(embedding.shape) == 2
        assert embedding.shape[-1] == self.speaker_embedding_size

        logits_prediction = self.compute_speaker_prediction(embedding)
        loss, prediction = self.loss_fn(logits_prediction, spk_label)

        self._log_train_acc(prediction, spk_label, batch_idx)
        self._log_train_loss(loss, batch_idx)

        return {"loss": loss}

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

    def _log_train_loss(self, loss: t.Tensor, batch_idx: int):
        self.metric_train_loss(loss)

        if batch_idx % 100 == 0:
            self.log(
                "train_loss",
                self.metric_train_loss.compute(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            self.metric_train_loss.reset()

    def validation_step(
        self,
        batch: SpeakerRecognitionBatch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ):
        assert isinstance(batch, SpeakerRecognitionBatch)

        audio_input = batch.audio_tensor
        label = batch.id_tensor

        embedding = self.compute_speaker_embedding(audio_input)

        assert len(embedding.shape) == 2
        assert embedding.shape[-1] == self.speaker_embedding_size

        logits_prediction = self.compute_speaker_prediction(embedding)
        loss, prediction = self.loss_fn(logits_prediction, label)

        self.metric_valid_acc(prediction, label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self.log_dict(
            {
                "val_acc": self.metric_valid_acc.compute(),
            },
            on_epoch=True,
            prog_bar=True,
        )
        self.metric_valid_acc.reset()

    def test_step(
        self,
        batch: SpeakerRecognitionBatch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ):
        assert isinstance(batch, SpeakerRecognitionBatch)

        if batch.batch_size != 1:
            raise ValueError("expecting a batch size of 1 for evaluation")

        audio_input = batch.audio_tensor
        sample_id = batch.keys

        embedding = self.compute_speaker_embedding(audio_input)

        if (
            len(embedding.shape) == 1
            and embedding.shape[0] == self.speaker_embedding_size
        ):
            embedding = embedding[None, :]

        assert len(embedding.shape) == 2
        assert embedding.shape[0] == batch.batch_size
        assert embedding.shape[1] == self.speaker_embedding_size

        embedding = t.stack([embedding.detach().to("cpu")])

        return {
            "embedding": embedding,
            "sample_id": sample_id,
        }

    def test_epoch_end(self, outputs: List[Any]) -> None:
        if len(self.test_pairs) == 1:
            outputs = [outputs]

        result_dict = {}

        for idx in range(len(outputs)):
            key = self.test_names[idx]

            results = evaluate_embeddings(
                self.evaluator, outputs[idx], self.test_pairs[idx], True
            )

            result_dict[f"test_eer_{key}"] = results["eer"]
            result_dict[f"test_eer_threshold_{key}"] = results["eer_threshold"]

        self.log_dict(result_dict)


########################################################################################
# utility methods


def evaluate_embeddings(
    evaluator: SpeakerRecognitionEvaluator,
    outputs: List[dict],
    pairs: List[SpeakerTrial],
    print_info: bool,
):
    # outputs is a list of dictionary with at least keys:
    # 'embedding' -> tensor with shape [BATCH_SIZE, EMBEDDING_SIZE]
    # 'sample_id' -> list of keys with length BATCH_SIZE
    embedding_list = extract_embedding_sample_list(outputs)

    result = evaluator.evaluate(pairs, embedding_list, print_info=print_info)
    result = {k: t.Tensor([v]) for k, v in result.items()}

    return result


def extract_embedding_sample_list(outputs: List[dict]):
    embedding_list: List[EmbeddingSample] = []

    for d in outputs:
        embedding_tensor = d["embedding"]
        sample_id_list = d["sample_id"]

        for idx, sample_id in enumerate(sample_id_list):
            embedding_list.append(
                EmbeddingSample(
                    sample_id=sample_id,
                    embedding=embedding_tensor[idx, :].squeeze(),
                )
            )

    return embedding_list
