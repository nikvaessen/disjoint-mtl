########################################################################################
#
# Heads on top of a feature extractor for speech recognition.
#
# Author(s): Nik Vaessen
########################################################################################
import json
import pathlib
from typing import Union, Optional

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
    blank_token_idx: int

    # optional for setting bias values correctly
    blank_initial_bias: Optional[int] = None
    character_distribution_json: Optional[pathlib.Path] = None
    character_vocabulary_json: Optional[pathlib.Path] = None


class LinearHead(SpeechRecognitionHead):
    def __init__(
        self, cfg: LinearHeadConfig, representation_dim: int, classification_dim: int
    ):
        super().__init__()

        self.cfg = cfg
        self.classification_dim = classification_dim
        print(f"\n{self.classification_dim=} {0 <= self.classification_dim < 3=}")
        assert 0 <= self.classification_dim < 3

        self.classification_layer = t.nn.Linear(
            in_features=representation_dim,
            out_features=classification_dim,
        )

        self.classification_layer.bias = self._init_bias(self.classification_layer.bias)

    def _init_bias(self, previous_bias: t.Tensor):
        if self.cfg.character_vocabulary_json is not None:
            with self.cfg.character_vocabulary_json.open("r") as f:
                idx_to_char = json.load(f)["idx_to_char"]
                idx_to_char = {int(k): v for k, v in idx_to_char.items()}
        else:
            idx_to_char = None

        if self.cfg.character_distribution_json is not None:
            with self.cfg.character_distribution_json.open("r") as f:
                char_dist = json.load(f)
        else:
            char_dist = None

        if char_dist is not None and idx_to_char is not None:
            assert all(v in char_dist for v in idx_to_char.values())
            assert len(char_dist) == self.classification_dim

        def get_bias_value(idx):
            if (
                idx == self.cfg.blank_token_idx
                and self.cfg.blank_initial_bias is not None
            ):
                return self.cfg.blank_initial_bias
            elif char_dist is not None and idx_to_char is not None:
                return char_dist[idx_to_char[idx]]
            else:
                return previous_bias[idx]

        bias_values = [get_bias_value(i) for i in range(self.classification_dim)]

        return t.nn.Parameter(
            t.tensor(bias_values, dtype=previous_bias.dtype),
            requires_grad=True,
        )

    def compute_prediction(self, sequence: t.Tensor) -> t.Tensor:
        return self.classification_layer(sequence)


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
