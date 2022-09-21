########################################################################################
#
# Implement WavLM as a speaker recognition network.
#
# Author(s): Nik Vaessen
########################################################################################

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch as t

from omegaconf import DictConfig
from transformers import WavLMModel

from data_utility.pipe.types import SpeakerTrial
from src.networks.speaker_recognition_module import SpeakerRecognitionLightningModule
from src.util.torch import freeze_module, unfreeze_module


########################################################################################
# Extend the speaker recognition lightning module


@dataclass
class WavLMForSpeakerRecognitionConfig:
    # settings for wav2vec2 architecture
    huggingface_id: str
    reset_weights: bool

    # if enabled, gradient checkpointing slows down iteration speed but saves memory
    use_gradient_checkpointing: bool

    # freeze logic
    freeze_cnn: bool
    freeze_transformer: bool  # this also freezes projector and rel. pos. emb
    num_steps_freeze_cnn: Optional[int]
    num_steps_freeze_transformer: Optional[int]


########################################################################################
# heads


########################################################################################
# complete network


class WavLMForSpeakerRecognition(SpeakerRecognitionLightningModule):
    def __init__(
        self,
        root_hydra_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        num_speakers: int,
        validation_pairs: List[SpeakerTrial],
        test_pairs: List[List[SpeakerTrial]],
        test_names: List[str],
        cfg: WavLMForSpeakerRecognitionConfig,
    ):
        super().__init__(
            root_hydra_config,
            loss_fn_constructor,
            num_speakers,
            validation_pairs,
            test_pairs,
            test_names,
        )

        self.cfg = cfg

        self.wavlm: WavLMModel = WavLMModel.from_pretrained(self.cfg.huggingface_id)

        if self.cfg.use_gradient_checkpointing:
            self.wavlm.gradient_checkpointing_enable()

        if "base" in self.cfg.huggingface_id:
            self.embedding_size = 768
        elif "large" in self.cfg.huggingface_id:
            self.embedding_size = 1024
        else:
            raise ValueError("unable to determine embedding size}")

        self.head = t.nn.Linear(
            in_features=self.embedding_size, out_features=num_speakers
        )

        # freeze logic
        self._freeze_cnn = self.cfg.freeze_cnn
        self._freeze_transformer = self.cfg.freeze_transformer
        self._num_steps_frozen = 0

    @property
    def speaker_embedding_size(self):
        return self.embedding_size

    def compute_speaker_embedding(self, input_tensor: t.Tensor) -> t.Tensor:
        sequence = self.wavlm(input_tensor).last_hidden_state
        embedding = t.mean(sequence, dim=1)

        return embedding

    def compute_speaker_prediction(self, embedding_tensor: t.Tensor) -> t.Tensor:
        logits = self.head(embedding_tensor)

        return logits

    def on_train_start(self) -> None:
        if self._freeze_cnn:
            freeze_module(self.wavlm.feature_extractor)

        if self._freeze_transformer:
            freeze_module(self.wavlm.feature_projection)
            freeze_module(self.wavlm.encoder)

            if self.wavlm.masked_spec_embed is not None:
                freeze_module(self.wavlm.masked_spec_embed)
            if self.wavlm.adapter is not None:
                freeze_module(self.wavlm.adapter)

        self._num_steps_frozen = 0

    def on_after_backward(self) -> None:
        self._num_steps_frozen = +1

        if (
            self._freeze_cnn
            and self.cfg.num_steps_freeze_cnn is not None
            and self._num_steps_frozen >= self.cfg.num_steps_freeze_cnn
        ):
            self._freeze_cnn = False
            unfreeze_module(self.wavlm.feature_extractor)

        if (
            self._freeze_transformer
            and self.cfg.num_steps_freeze_transformer is not None
            and self._num_steps_frozen >= self.cfg.num_steps_freeze_transformer
        ):
            self._freeze_transformer = False
            unfreeze_module(self.wavlm.feature_projection)
            unfreeze_module(self.wavlm.encoder)

            if self.wavlm.masked_spec_embed is not None:
                unfreeze_module(self.wavlm.masked_spec_embed)
            if self.wavlm.adapter is not None:
                unfreeze_module(self.wavlm.adapter)
