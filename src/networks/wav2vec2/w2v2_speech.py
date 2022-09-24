########################################################################################
#
# Implement wav2vec2 as a speaker recognition network
#
# Author(s): Nik Vaessen
########################################################################################

from dataclasses import dataclass
from typing import Callable, List, Optional, Dict, Tuple

import torch as t

from omegaconf import DictConfig
from transformers import Wav2Vec2Model

from src.networks.heads import SpeechHeadConfig, construct_speech_head
from src.networks.speech_recognition_module import SpeechRecognitionLightningModule
from src.util.freeze import FreezeManager


########################################################################################
# Extend the speaker recognition lightning module


@dataclass
class Wav2vec2ForSpeechRecognitionConfig:
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
    head_cfg: SpeechHeadConfig


########################################################################################
# complete network


class Wav2vec2ForSpeechRecognition(SpeechRecognitionLightningModule):
    def __init__(
        self,
        root_hydra_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        idx_to_char: Dict[int, str],
        test_names: List[str],
        cfg: Wav2vec2ForSpeechRecognitionConfig,
    ):
        super().__init__(
            hyperparameter_config=root_hydra_config,
            loss_fn_constructor=loss_fn_constructor,
            idx_to_char=idx_to_char,
            test_names=test_names,
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

        self.head = construct_speech_head(
            self.cfg.head_cfg,
            representation_dim=self.embedding_size,
            classification_dim=self.vocab_size,
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
        self, wav_input: t.Tensor, lengths: List[int]
    ) -> Tuple[t.Tensor, List[int]]:
        # transform input
        # (of shape [BS, 1, NUM_AUDIO_SAMPLES] or [1, NUM_AUDIO_SAMPLES])
        # to the required [BS, NUM_AUDIO_SAMPLES]
        if len(wav_input.shape) == 3 and wav_input.shape[1] == 1:
            wav_input = t.squeeze(wav_input)
        if len(wav_input.shape) == 1:
            wav_input = t.stack([wav_input])

        # first compute the wav2vec2 embeddings:
        # will be shape [BS, NUM_WINDOWS, EMBEDDING_SIZE]
        num_audio_samples = [wav_input.shape[1] for _ in range(wav_input.shape[0])]
        (
            audio_features,
            num_audio_features,
            attention_mask,
        ) = self._extract_features(wav_input, num_audio_samples)

        wav2vec2_embeddings = self._compute_wav2vec2_embeddings(
            audio_features, attention_mask
        )

        # we end with all the operations to get to the speaker embeddings
        return wav2vec2_embeddings, num_audio_features

    def _extract_features(self, wav_input: t.Tensor, num_audio_samples: List[int]):
        # wav input should be of shape [BATCH_SIZE, NUM_AUDIO_SAMPLES]

        # first compute audio features with CNN
        features = self.wav2vec2.feature_extractor(wav_input)
        features = features.transpose(1, 2)

        # project channels of CNN output into a sequence of input token embeddings
        features, _ = self.wav2vec2.feature_projection(features)
        num_feature_tokens = self._compute_feature_extractor_lengths(num_audio_samples)
        attention_mask = self._construct_attention_mask(
            num_audio_samples, max(num_feature_tokens), device=wav_input.device
        )

        # optionally apply masking to sequence (in time and feature axis)
        features = self.wav2vec2._mask_hidden_states(
            features, attention_mask=attention_mask
        )

        # features should be of shape [BATCH_SIZE, NUM_FRAMES, NUM_FEATURES]
        bs, num_frames, num_features = features.shape
        assert bs == wav_input.shape[0]
        assert num_frames == max(num_feature_tokens)
        assert num_features == self.embedding_size

        return features, num_feature_tokens, attention_mask

    def _compute_wav2vec2_embeddings(
        self, input_token_sequence: t.Tensor, attention_mask: t.Tensor = None
    ):
        # input token sequence is of shape [BATCH_SIZE, NUM_FRAMES, NUM_FEATURES]
        # optional attention mask is of shape [BATCH_SIZE, NUM_FRAMES], where
        # 1 means `pay attention` and 0 means `skip processing this frame`.
        encoder_output = self.wav2vec2.encoder(
            input_token_sequence, attention_mask=attention_mask
        )

        embedding = encoder_output.last_hidden_state

        # embedding should be of shape [BATCH_SIZE, NUM_FRAMES, NUM_FEATURES]
        bs, num_frames, num_features = embedding.shape
        assert bs == input_token_sequence.shape[0] == attention_mask.shape[0]
        assert num_frames == input_token_sequence.shape[1] == attention_mask.shape[1]
        assert num_features == self.embedding_size

        return embedding

    def _construct_attention_mask(
        self, num_audio_samples: List[int], feature_sequence_length: int, device: str
    ):
        assert len(num_audio_samples) >= 1

        # init assumes all tokens are attended to
        bs = len(num_audio_samples)
        max_num_audio_samples = max(num_audio_samples)
        attention_mask = t.ones((bs, max_num_audio_samples), dtype=t.long)

        for idx, length in enumerate(num_audio_samples):
            assert length >= 0

            # set each token which is 'padding' to 0
            attention_mask[idx, length:] = 0

        attention_mask = self.wav2vec2._get_feature_vector_attention_mask(
            feature_sequence_length, attention_mask
        )

        return attention_mask.to(device=device)

    def _compute_feature_extractor_lengths(self, num_audio_samples: List[int]):
        num_feature_lengths = self.wav2vec2._get_feat_extract_output_lengths(
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
