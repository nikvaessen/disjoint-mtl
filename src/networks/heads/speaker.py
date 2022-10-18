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
class LinearProjectionHeadConfig(CastingConfig):
    # settings related to architecture
    use_projection_layer: bool
    projection_layer_dim: int

    # settings related to loss-function
    use_cosine_linear: bool  # set to true when using aam-softmax loss

    # regularization settings
    drop_prob: float

    # amount of frames in embedding sequence to use while in train mode.
    enable_train_chunk: bool
    train_random_chunk_size: int  # ~25 ms per frame. 40==1 second chunk size


class LinearProjectionHead(SpeakerRecognitionHead):
    def __init__(
        self,
        cfg: LinearProjectionHeadConfig,
        representation_dim: int,
        classification_dim: int,
    ):
        super().__init__()

        self.cfg = cfg

        if self.cfg.use_projection_layer:
            self.projection_layer = t.nn.Sequential(
                t.nn.Linear(
                    in_features=representation_dim,
                    out_features=self.cfg.projection_layer_dim,
                ),
                t.nn.Dropout(p=self.cfg.drop_prob),
                t.nn.LeakyReLU(),
            )
        else:
            self.projection_layer = t.nn.Identity()

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
        num_frames = projected_sequence.shape[1]

        if (
            self.training
            and self.cfg.enable_train_chunk
            and num_frames > self.cfg.train_random_chunk_size
        ):
            min_idx = 0
            max_idx = num_frames - self.cfg.train_random_chunk_size - 1

            start_idx = t.randint(low=min_idx, high=max_idx, size=())
            stop_idx = start_idx + self.cfg.train_random_chunk_size

            embedding = t.mean(projected_sequence[:, start_idx:stop_idx, :], dim=1)
        else:
            embedding = t.mean(projected_sequence, dim=1)

        return embedding

    @property
    def speaker_embedding_size(self):
        return self.cfg.projection_layer_dim


########################################################################################
# x-vector as head


@dataclass
class XvectorHeadConfig(CastingConfig):
    # settings related to loss-function
    use_cosine_linear: bool  # set to true when using aam-softmax loss

    classifier_cfg: LinearProjectionHeadConfig


class XvectorHead(SpeakerRecognitionHead):
    def __init__(
        self, cfg: XvectorHeadConfig, representation_dim: int, classification_dim: int
    ):
        super().__init__()

        self.cfg = cfg
        self.cfg.classifier_cfg.use_cosine_linear = self.cfg.use_cosine_linear

        self.xvector = xvector.Xvector(in_channels=representation_dim)
        # remove stat pooling and linear layer at the end of xvector
        del self.xvector.blocks[-2:]

        self.classifier = LinearProjectionHead(
            self.cfg.classifier_cfg, 1500, classification_dim
        )

    def compute_prediction(self, embedding: t.Tensor) -> t.Tensor:
        return self.classifier.compute_prediction(embedding)

    def compute_embedding(self, sequence: t.Tensor) -> t.Tensor:
        tdnn_output = self.xvector(sequence)
        embedding = self.classifier.compute_embedding(tdnn_output)
        return embedding

    @property
    def speaker_embedding_size(self):
        return self.classifier.speaker_embedding_size


########################################################################################
# ECAPA-TDNN as head


@dataclass
class EcapaTdnnHeadConfig(CastingConfig):
    # settings related to loss-function
    use_cosine_linear: bool  # set to true when using aam-softmax loss


class EcapaTdnnHead(SpeakerRecognitionHead):
    def __init__(
        self, cfg: EcapaTdnnHeadConfig, representation_dim: int, classification_dim: int
    ):
        super().__init__()

        self.cfg = cfg

        self.ecapa = ecapa.ECAPA_TDNN(input_size=representation_dim)

        if self.cfg.use_cosine_linear:
            self.classification_layer = CosineLinear(
                in_features=192,
                out_features=classification_dim,
            )
        else:
            self.classification_layer = t.nn.Linear(
                in_features=192,
                out_features=classification_dim,
            )

    def compute_prediction(self, embedding: t.Tensor) -> t.Tensor:
        return self.classification_layer(embedding)

    def compute_embedding(self, sequence: t.Tensor) -> t.Tensor:
        embedding = self.ecapa(sequence)

        if len(embedding.shape) == 3 and embedding.shape[1] == 1:
            embedding = t.squeeze(embedding, dim=1)

        return embedding

    @property
    def speaker_embedding_size(self):
        return 192


########################################################################################
# Encapsulate all heads

SpeakerHeadConfig = Union[LinearProjectionHeadConfig, XvectorHeadConfig, EcapaTdnnHead]


def construct_speaker_head(
    cfg: SpeakerHeadConfig, representation_dim: int, classification_dim: int
) -> SpeakerRecognitionHead:
    if isinstance(cfg, LinearProjectionHeadConfig):
        return LinearProjectionHead(
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
