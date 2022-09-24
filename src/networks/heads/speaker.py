########################################################################################
#
# Heads on top of a feature extractor for speaker recognition.
#
# Author(s): Nik Vaessen
########################################################################################
from typing import Union

from abc import abstractmethod
from dataclasses import dataclass

import speechbrain.lobes.models.ECAPA_TDNN as ecapa
import speechbrain.lobes.models.Xvector as xvector

import torch as t

from src.layers.cosine_linear import CosineLinear
from src.util.config_util import CastingConfig

########################################################################################
# abstract


class SpeakerRecognitionHead(t.nn.Module):
    def forward(self, sequence: t.Tensor):
        embedding = self.compute_embedding(sequence)
        prediction = self.compute_prediction(embedding)

        return embedding, prediction

    @abstractmethod
    def compute_prediction(self, embedding: t.Tensor) -> t.Tensor:
        pass

    @abstractmethod
    def compute_embedding(self, sequence: t.Tensor) -> t.Tensor:
        pass

    @property
    @abstractmethod
    def speaker_embedding_size(self):
        pass


########################################################################################
# Simple FC network layer as head


@dataclass
class LinearHeadConfig(CastingConfig):
    projection_layer_dim: int
    drop_prob: float
    use_cosine_linear: bool  # set to true when using aam-softmax loss


class LinearHead(SpeakerRecognitionHead):
    def __init__(
        self, cfg: LinearHeadConfig, representation_dim: int, classification_dim: int
    ):
        super().__init__()

        self.cfg = cfg

        self.projection_layer = t.nn.Sequential(
            t.nn.Linear(
                in_features=representation_dim,
                out_features=self.cfg.projection_layer_dim,
            ),
            t.nn.Dropout(p=self.cfg.drop_prob),
        )

        if self.cfg.use_cosine_linear:
            self.classification_layer = CosineLinear(
                in_features=self.cfg.projection_layer_dim,
                out_features=classification_dim,
            )
        else:
            self.classification_layer = t.nn.Linear(
                in_features=self.cfg.projection_layer_dim,
                out_features=classification_dim,
            )

    def compute_prediction(self, embedding: t.Tensor) -> t.Tensor:
        return self.classification_layer(embedding)

    def compute_embedding(self, sequence: t.Tensor) -> t.Tensor:
        projected_sequence = self.projection_layer(sequence)
        embedding = t.mean(projected_sequence, dim=1)

        return embedding

    @property
    def speaker_embedding_size(self):
        return self.cfg.projection_layer_dim


########################################################################################
# x-vector as head


@dataclass
class XvectorHeadConfig(CastingConfig):
    use_cosine_linear: bool  # set to true when using aam-softmax loss


class XvectorHead(SpeakerRecognitionHead):
    def __init__(
        self, cfg: XvectorHeadConfig, representation_dim: int, classification_dim: int
    ):
        super().__init__()

        self.cfg = cfg

        self.xvector = xvector.Xvector(in_channels=representation_dim)
        # self.classifier = ...

    def compute_prediction(self, embedding: t.Tensor) -> t.Tensor:
        raise NotImplemented()

    def compute_embedding(self, sequence: t.Tensor) -> t.Tensor:
        raise NotImplemented()


########################################################################################
# ECAPA-TDNN as head


@dataclass()
class EcapaTdnnHeadConfig(CastingConfig):
    use_cosine_linear: bool  # set to true when using aam-softmax loss


class EcapaTdnnHead(SpeakerRecognitionHead):
    def __init__(
        self, cfg: EcapaTdnnHeadConfig, representation_dim: int, classification_dim: int
    ):
        super().__init__()

        self.cfg = cfg

    def compute_prediction(self, embedding: t.Tensor) -> t.Tensor:
        raise NotImplemented()

    def compute_embedding(self, sequence: t.Tensor) -> t.Tensor:
        raise NotImplemented()


########################################################################################
# Encapsulate all heads

SpeakerHeadConfig = Union[LinearHeadConfig, XvectorHeadConfig, EcapaTdnnHead]


def construct_speaker_head(
    cfg: SpeakerHeadConfig, representation_dim: int, classification_dim: int
) -> SpeakerRecognitionHead:
    if isinstance(cfg, LinearHeadConfig):
        return LinearHead(
            cfg,
            representation_dim=representation_dim,
            classification_dim=classification_dim,
        )
    elif isinstance(cfg, XvectorHeadConfig):
        return XvectorHead(
            cfg,
            representation_dim=representation_dim,
            classification_dim=classification_dim,
        )
    elif isinstance(cfg, EcapaTdnnHeadConfig):
        return EcapaTdnnHead(
            cfg,
            representation_dim=representation_dim,
            classification_dim=classification_dim,
        )
    else:
        raise ValueError(f"unknown {cfg=}")
