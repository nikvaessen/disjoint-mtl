########################################################################################
#
# Heads on top of a feature extractor for speaker recognition.
#
# Author(s): Nik Vaessen
########################################################################################

from typing import Union, Optional

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

    # pooling method
    pool_method: str

    # settings related to loss-function
    use_cosine_linear: bool  # set to true when using aam-softmax loss

    # amount of frames in embedding sequence to use while in train mode and pooling method is "mean_chunked.
    mean_random_chunk_size: Optional[int] = None  # ~25 ms per frame.

    # settings related to projection layer, if enabled
    projection_layer_dim: Optional[int] = None
    projection_layer_dim_drop_prob: Optional[float] = None


class LinearProjectionHead(SpeakerRecognitionHead):
    pool_methods = ["mean", "first", "mean_chunked"]

    def __init__(
        self,
        cfg: LinearProjectionHeadConfig,
        representation_dim: int,
        classification_dim: int,
    ):
        super().__init__()

        self.cfg = cfg

        if self.cfg.pool_method not in self.pool_methods:
            raise ValueError(f"{self.cfg.pool_method} not int {self.pool_methods}")

        self._embedding_size = (
            self.cfg.projection_layer_dim
            if self.cfg.use_projection_layer
            else representation_dim
        )

        if self.cfg.use_projection_layer:
            self.projection_layer = t.nn.Sequential(
                t.nn.Linear(
                    in_features=representation_dim,
                    out_features=self.cfg.projection_layer_dim,
                ),
                t.nn.Dropout(p=self.cfg.projection_layer_dim_drop_prob),
                t.nn.LeakyReLU(),
            )
        else:
            self.projection_layer = t.nn.Identity()

        if self.cfg.use_cosine_linear:
            self.classification_layer = CosineLinear(
                in_features=self._embedding_size,
                out_features=classification_dim,
            )
        else:
            self.classification_layer = t.nn.Linear(
                in_features=self._embedding_size,
                out_features=classification_dim,
            )

    def compute_prediction(self, embedding: t.Tensor) -> t.Tensor:
        return self.classification_layer(embedding)

    def pool_sequence(self, sequence: t.Tensor, dim: int = 1):
        assert len(sequence.shape) == 3
        assert 0 <= dim < 3

        num_frames = sequence.shape[dim]

        if self.cfg.pool_method == "mean" or self.cfg.pool_method == "mean_chunked":
            if (
                self.training
                and self.cfg.pool_method == "mean_chunked"
                and num_frames > self.cfg.train_random_chunk_size
            ):
                min_idx = 0
                max_idx = num_frames - self.cfg.train_random_chunk_size - 1

                start_idx = t.randint(low=min_idx, high=max_idx, size=())
                stop_idx = start_idx + self.cfg.train_random_chunk_size

                if dim == 0:
                    sequence = sequence[start_idx:stop_idx, :, :]
                elif dim == 1:
                    sequence = sequence[:, start_idx:stop_idx, :]
                else:
                    sequence = sequence[:, :, start_idx:stop_idx]

            embedding = t.mean(sequence, dim=dim)

        elif self.cfg.pool_method == "first":
            if dim == 0:
                embedding = sequence[0, :, :]
            elif dim == 1:
                embedding = sequence[:, 0, :]
            else:
                embedding = sequence[:, :, 0]
        else:
            raise ValueError(f"unknown {self.cfg.pool_method=}")

        return embedding

    def compute_embedding(self, sequence: t.Tensor) -> t.Tensor:
        projected_sequence = self.projection_layer(sequence)
        embedding = self.pool_sequence(projected_sequence)

        return embedding

    @property
    def speaker_embedding_size(self):
        return self._embedding_size


########################################################################################
# x-vector as head


@dataclass
class XvectorHeadConfig(CastingConfig):
    # settings related to loss-function
    use_cosine_linear: bool  # set to true when using aam-softmax loss


class XvectorHead(SpeakerRecognitionHead):
    def __init__(
        self, cfg: XvectorHeadConfig, representation_dim: int, classification_dim: int
    ):
        super().__init__()

        self.cfg = cfg

        self.xvector = xvector.Xvector(
            in_channels=representation_dim,
        )

        if self.cfg.use_cosine_linear:
            self.classification_layer = CosineLinear(
                in_features=512,
                out_features=classification_dim,
            )
        else:
            self.classification_layer = t.nn.Linear(
                in_features=5125,
                out_features=classification_dim,
            )

    def compute_prediction(self, embedding: t.Tensor) -> t.Tensor:
        return self.classification_layer(embedding)

    def compute_embedding(self, sequence: t.Tensor) -> t.Tensor:
        embedding = self.xvector(sequence)

        if len(embedding.shape) == 3 and embedding.shape[1] == 1:
            embedding = t.squeeze(embedding, dim=1)

        return embedding

    @property
    def speaker_embedding_size(self):
        return 512


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
