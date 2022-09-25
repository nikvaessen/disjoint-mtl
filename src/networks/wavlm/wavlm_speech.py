########################################################################################
#
# Implement wavlm as a speaker recognition network
#
# Author(s): Nik Vaessen
########################################################################################

from dataclasses import dataclass
from typing import Callable, List, Optional, Dict, Tuple

import torch as t

from omegaconf import DictConfig
from transformers import WavLMModel

from src.networks.heads import SpeechHeadConfig, construct_speech_head
from src.networks.speech_recognition_module import SpeechRecognitionLightningModule
from src.util.freeze import FreezeManager


########################################################################################
# Extend the speaker recognition lightning module


@dataclass
class WavLMForSpeechRecognitionConfig:
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

    # head on top of wavLM for speech recognition
    head_cfg: SpeechHeadConfig


########################################################################################
# complete network


class WavLMForSpeechRecognition(SpeechRecognitionLightningModule):
    def __init__(
        self,
        root_hydra_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        idx_to_char: Dict[int, str],
        test_names: List[str],
        cfg: WavLMForSpeechRecognitionConfig,
    ):
        super().__init__(
            hyperparameter_config=root_hydra_config,
            loss_fn_constructor=loss_fn_constructor,
            idx_to_char=idx_to_char,
            test_names=test_names,
        )

        self.cfg = cfg
        self.vocab_size = len(idx_to_char)

        self.wavlm: WavLMModel = WavLMModel.from_pretrained(self.cfg.huggingface_id)

        if self.cfg.use_gradient_checkpointing:
            self.wavlm.gradient_checkpointing_enable()

        if "base" in self.cfg.huggingface_id:
            self.embedding_size = 768
        elif "large" in self.cfg.huggingface_id:
            self.embedding_size = 1024
        else:
            raise ValueError("unable to determine embedding size}")

        self.head = construct_speech_head(
            self.cfg.head_cfg,
            representation_dim=self.embedding_size,
            classification_dim=self.vocab_size,
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

    def compute_embedding_sequence(
        self, input_tensor: t.Tensor, lengths: List[int]
    ) -> Tuple[t.Tensor, List[int]]:
        sequence_output = self.wavlm(
            input_values=input_tensor,
            attention_mask=self._construct_attention_mask(
                num_audio_samples=lengths,
                device=self.device,
            ),
        ).last_hidden_state
        sequence_lengths = self._compute_feature_extractor_lengths(lengths)

        return sequence_output, sequence_lengths

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
        num_feature_lengths = self.wavlm._get_feat_extract_output_lengths(
            t.LongTensor(num_audio_samples)
        ).tolist()

        return num_feature_lengths

    def compute_vocabulary_prediction(
        self, embedding_tensor: t.Tensor, lengths: List[int]
    ) -> Tuple[t.Tensor, List[int]]:
        letter_prediction = self.head(embedding_tensor)

        return letter_prediction, lengths

    def on_train_start(self) -> None:
        self.freeze_cnn.on_train_start()
        self.freeze_transformer.on_train_start()

    def on_after_backward(self) -> None:
        self.freeze_cnn.on_after_backward()
        self.freeze_transformer.on_after_backward()
