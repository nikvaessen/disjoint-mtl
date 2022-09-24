########################################################################################
#
# Heads on top of a feature extractor for speech recognition.
#
# Author(s): Nik Vaessen
########################################################################################

from typing import Union

from abc import abstractmethod
from dataclasses import dataclass

import torch as t

from src.util.config_util import CastingConfig

########################################################################################
# abstract


class SpeechRecognitionHead(t.nn.Module):
    def forward(self, sequence: t.Tensor):
        prediction = self.compute_prediction(sequence)

        return prediction

    @abstractmethod
    def compute_prediction(self, embedding: t.Tensor) -> t.Tensor:
        pass


########################################################################################
# Simple FC network layer as head


@dataclass
class LinearHeadConfig(CastingConfig):
    pass


class LinearHead(SpeechRecognitionHead):
    def __init__(
        self, cfg: LinearHeadConfig, representation_dim: int, classification_dim: int
    ):
        super().__init__()

        self.cfg = cfg

        self.classification_layer = t.nn.Sequential(
            t.nn.Linear(
                in_features=representation_dim,
                out_features=classification_dim,
            )
        )

    def compute_prediction(self, embedding: t.Tensor) -> t.Tensor:
        return self.classification_layer(embedding)


########################################################################################
# Encapsulate all heads

SpeechHeadConfig = Union[LinearHeadConfig]


def construct_speech_head(
    cfg: SpeechHeadConfig, representation_dim: int, classification_dim: int
) -> SpeechRecognitionHead:
    if isinstance(cfg, LinearHeadConfig):
        return LinearHead(
            cfg,
            representation_dim=representation_dim,
            classification_dim=classification_dim,
        )
    else:
        raise ValueError(f"unknown {cfg=}")
