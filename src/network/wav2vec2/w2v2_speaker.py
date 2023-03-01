########################################################################################
#
# Implement wav2vec2 as a speaker recognition network
#
# Author(s): Nik Vaessen
########################################################################################

from dataclasses import dataclass
from typing import Callable, List, Optional, Any

import torch as t

from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import Wav2Vec2Model

from data_utility.eval.speaker.evaluator import SpeakerTrial
from src.network.heads import SpeakerHeadConfig, construct_speaker_head
from src.network.speaker_recognition_module import SpeakerRecognitionLightningModule
from src.optim.loss import AngularAdditiveMarginSoftMaxLoss
from src.util.freeze import FreezeManager


########################################################################################
# Extend the speaker recognition lightning module


@dataclass
class Wav2vec2ForSpeakerRecognitionConfig:
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

    # head on top of wav2vec2 for speaker recognition
    head_cfg: SpeakerHeadConfig
    head_layer: int = -1 # on which layer to apply the speaker recognition head (-1=last)

    # reg settings
    apply_spec_augment: bool = True

    mask_time_prob: int = 0.05
    mask_time_length: int = 10
    mask_time_min_masks: int = 2
    mask_feature_prob: float = 0
    mask_feature_length: int = 10
    mask_feature_min_masks: int = 0

    hidden_dropout: int = 0.1
    attention_dropout: int = 0.1
    feat_proj_dropout: int = 0.1


########################################################################################
# complete network


class Wav2vec2ForSpeakerRecognition(SpeakerRecognitionLightningModule):
    def __init__(
        self,
        root_hydra_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        num_speakers: int,
        test_pairs: List[List[SpeakerTrial]],
        test_names: List[str],
        cfg: Wav2vec2ForSpeakerRecognitionConfig,
    ):
        super().__init__(
            root_hydra_config,
            loss_fn_constructor,
            num_speakers,
            test_pairs,
            test_names,
        )

        self.cfg = cfg

        self.wav2vec2: Wav2Vec2Model = Wav2Vec2Model.from_pretrained(
            self.cfg.huggingface_id,
            **{
                "apply_spec_augment": cfg.apply_spec_augment,
                "mask_time_prob": cfg.mask_time_prob,
                "mask_time_length": cfg.mask_time_length,
                "mask_time_min_masks": cfg.mask_time_min_masks,
                "mask_feature_prob": cfg.mask_feature_prob,
                "mask_feature_length": cfg.mask_feature_length,
                "mask_feature_min_masks": cfg.mask_feature_min_masks,
                "hidden_dropout": cfg.hidden_dropout,
                "attention_dropout": cfg.attention_dropout,
                "feat_proj_dropout": cfg.feat_proj_dropout,
            },
        )

        if self.cfg.use_gradient_checkpointing:
            self.wav2vec2.gradient_checkpointing_enable()

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
            loss_fn=self.loss_fn,
        )

        # freeze logic
        self.freeze_cnn = FreezeManager(
            module=self.wav2vec2.feature_extractor,
            is_frozen_at_init=self.cfg.freeze_cnn,
            num_steps_frozen=self.cfg.num_steps_freeze_cnn,
        )
        self.freeze_transformer = FreezeManager(
            module=[
                x
                for x in (
                    self.wav2vec2.feature_projection,
                    self.wav2vec2.encoder,
                    self.wav2vec2.masked_spec_embed
                    if hasattr(self.wav2vec2, "masked_spec_embed")
                    else None,
                    self.wav2vec2.adapter,
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
        result = self.wav2vec2(input_tensor)

        if self.cfg.head_layer == -1:
            sequence = result.last_hidden_state
        else:
            sequence = result.hidden_states[self.cfg.head_layer]

        embedding = self.head.compute_embedding(sequence)
        return embedding

    def compute_speaker_prediction(self, embedding_tensor: t.Tensor) -> t.Tensor:
        logits = self.head.compute_prediction(embedding_tensor)

        return logits

    def on_train_start(self) -> None:
        self.freeze_cnn.on_train_start()
        self.freeze_transformer.on_train_start()

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        self.freeze_cnn.on_train_batch_end()
        self.freeze_transformer.on_train_batch_end()
