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

from data_utility.eval.speaker.evaluator import SpeakerTrial
from src.networks.heads import SpeakerHeadConfig, construct_speaker_head
from src.networks.speaker_recognition_module import SpeakerRecognitionLightningModule
from src.util.freeze import FreezeManager


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

    # head on top of wavLM for speaker recognition
    head_cfg: SpeakerHeadConfig


########################################################################################
# complete network


class WavLMForSpeakerRecognition(SpeakerRecognitionLightningModule):
    def __init__(
        self,
        root_hydra_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        num_speakers: int,
        test_pairs: List[List[SpeakerTrial]],
        test_names: List[str],
        cfg: WavLMForSpeakerRecognitionConfig,
    ):
        super().__init__(
            root_hydra_config,
            loss_fn_constructor,
            num_speakers,
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

        self.head = construct_speaker_head(
            self.cfg.head_cfg,
            representation_dim=self.embedding_size,
            classification_dim=self.num_speakers,
        )

        # freeze logic
        self.freeze_cnn = FreezeManager(
            module=self.wavlm.feature_extractor,
            is_frozen_at_init=self.cfg.freeze_cnn,
            num_steps_frozen=self.cfg.num_steps_freeze_cnn,
        )
        self.freeze_transformer = FreezeManager(
            module=[
                x
                for x in (
                    self.wavlm.feature_projection,
                    self.wavlm.encoder,
                    self.wavlm.masked_spec_embed,
                    self.wavlm.adapter,
                )
                if x is not None
            ],
            is_frozen_at_init=self.cfg.freeze_transformer,
            num_steps_frozen=self.cfg.num_steps_freeze_transformer,
        )

    @property
    def speaker_embedding_size(self):
        return self.head.speaker_embedding_size

    def compute_speaker_embedding(self, input_tensor: t.Tensor) -> t.Tensor:
        sequence = self.wavlm(input_tensor).last_hidden_state
        embedding = self.head.compute_embedding(sequence)

        return embedding

    def compute_speaker_prediction(self, embedding_tensor: t.Tensor) -> t.Tensor:
        logits = self.head.compute_prediction(embedding_tensor)

        return logits

    def on_train_start(self) -> None:
        self.freeze_cnn.on_train_start()
        self.freeze_transformer.on_train_start()

    def on_after_backward(self) -> None:
        self.freeze_cnn.on_after_backward()
        self.freeze_transformer.on_after_backward()
