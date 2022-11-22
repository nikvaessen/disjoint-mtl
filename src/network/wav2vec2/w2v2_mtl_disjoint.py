########################################################################################
#
# Implement wav2vec2 as a speech recognition network.
#
# Author(s): Nik Vaessen
########################################################################################

from dataclasses import dataclass
from typing import Callable, List, Optional, Dict, Tuple, Iterator

import torch as t

from omegaconf import DictConfig
from torch.nn import Parameter
from torch.optim import Optimizer
from transformers import Wav2Vec2Model

from data_utility.eval.speaker.evaluator import SpeakerTrial
from src.network.disjoint_mtl_recognition_module import DisjointMTLLightningModule
from src.network.heads import (
    SpeechHeadConfig,
    construct_speech_head,
    construct_speaker_head,
    SpeakerHeadConfig,
)
from src.util.freeze import FreezeManager


########################################################################################
# Extend the speaker recognition lightning module


@dataclass
class Wav2vec2ForDisjointMTLConfig:
    # settings for wav2vec2 architecture
    huggingface_id: str
    reset_weights: bool

    # if enabled, gradient checkpointing slows down iteration speed but saves memory
    use_gradient_checkpointing: bool

    # opt settings (conflict-adverse grad descent)
    apply_ca_grad: bool
    ca_grad_c: 0.5

    # freeze logic
    freeze_cnn: bool
    freeze_transformer: bool  # this also freezes projector and rel. pos. emb
    num_steps_freeze_cnn: Optional[int]
    num_steps_freeze_transformer: Optional[int]

    # head on top of wav2vec2 for speaker recognition
    speech_head_cfg: SpeechHeadConfig
    speaker_head_cfg: SpeakerHeadConfig


########################################################################################
# complete network


class Wav2vec2ForDisjointMTL(DisjointMTLLightningModule):
    def __init__(
        self,
        root_hydra_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        idx_to_char: Dict[int, str],
        num_speakers: int,
        val_names: List[str],
        val_modes: List[str],
        test_pairs: List[List[SpeakerTrial]],
        test_names: List[str],
        test_modes: List[str],
        cfg: Wav2vec2ForDisjointMTLConfig,
    ):
        super().__init__(
            hyperparameter_config=root_hydra_config,
            loss_fn_constructor=loss_fn_constructor,
            num_speakers=num_speakers,
            idx_to_char=idx_to_char,
            val_names=val_names,
            val_modes=val_modes,
            test_names=test_names,
            test_pairs=test_pairs,
            test_modes=test_modes,
            apply_ca_grad=cfg.apply_ca_grad,
            ca_grad_c=cfg.ca_grad_c,
        )

        self.cfg = cfg
        self.vocab_size = len(idx_to_char)

        self.wav2vec2: Wav2Vec2Model = Wav2Vec2Model.from_pretrained(
            self.cfg.huggingface_id
        )

        if self.cfg.use_gradient_checkpointing:
            self.wav2vec2.gradient_checkpointing_enable()

        if "base" in self.cfg.huggingface_id:
            self.embedding_size = 768
        elif "large" in self.cfg.huggingface_id:
            self.embedding_size = 1024
        else:
            raise ValueError("unable to determine embedding size}")

        self.speech_head = construct_speech_head(
            self.cfg.speech_head_cfg,
            representation_dim=self.embedding_size,
            classification_dim=self.vocab_size,
        )
        self.speaker_head = construct_speaker_head(
            self.cfg.speaker_head_cfg,
            representation_dim=self.embedding_size,
            classification_dim=self.num_speakers,
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
                    self.wav2vec2.masked_spec_embed,
                    self.wav2vec2.adapter,
                )
                if x is not None
            ],
            is_frozen_at_init=self.cfg.freeze_transformer,
            num_steps_frozen=self.cfg.num_steps_freeze_transformer,
        )

    def compute_embedding_sequence(
        self, input_tensor: t.Tensor, lengths: List[int]
    ) -> Tuple[t.Tensor, List[int]]:
        sequence_output = self.wav2vec2(
            input_values=input_tensor,
            attention_mask=self._construct_attention_mask(
                num_audio_samples=lengths,
                device=self.device,
            ),
        ).last_hidden_state
        sequence_lengths = self._compute_feature_extractor_lengths(lengths)

        return sequence_output, sequence_lengths

    def shared_params(self) -> Iterator[Tuple[str, Parameter]]:
        return self.wav2vec2.named_parameters()

    def _construct_attention_mask(self, num_audio_samples: List[int], device: str):
        assert len(num_audio_samples) >= 1

        # init assumes all tokens are attended to
        bs = len(num_audio_samples)
        max_num_audio_samples = max(num_audio_samples)
        attention_mask = t.ones((bs, max_num_audio_samples), dtype=t.long)

        for idx, length in enumerate(num_audio_samples):
            assert length >= 0

            # set each token which is 'padding' to 0
            attention_mask[idx, length:] = 0

        return attention_mask.to(device=device)

    def _compute_feature_extractor_lengths(self, num_audio_samples: List[int]):
        num_feature_lengths = self.wav2vec2._get_feat_extract_output_lengths(
            t.LongTensor(num_audio_samples)
        ).tolist()

        return num_feature_lengths

    def compute_vocabulary_prediction(
        self, embedding_tensor: t.Tensor, lengths: List[int]
    ) -> Tuple[t.Tensor, List[int]]:
        letter_prediction = self.speech_head(embedding_tensor)

        return letter_prediction, lengths

    @property
    def speaker_embedding_size(self):
        return self.speaker_head.speaker_embedding_size

    def compute_speaker_prediction(self, embedding_tensor: t.Tensor) -> t.Tensor:
        return self.speaker_head.compute_prediction(embedding_tensor)

    def compute_speaker_embedding(self, sequence_tensor: t.Tensor) -> t.Tensor:
        return self.speaker_head.compute_embedding(sequence_tensor)

    def on_train_start(self) -> None:
        self.freeze_cnn.on_train_start()
        self.freeze_transformer.on_train_start()

    def on_before_optimizer_step(
        self, optimizer: Optimizer, optimizer_idx: int
    ) -> None:
        self.freeze_cnn.on_before_optimizer_step()
        self.freeze_transformer.on_before_optimizer_step()
