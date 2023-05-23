########################################################################################
#
# Common types for encapsulating data.
#
# Author(s): Nik Vaessen
########################################################################################

from typing import Optional, List, Dict
from dataclasses import dataclass

import torch as t

from data_utility.eval.speech.transform import encode_transcription
from data_utility.pipe.pad import collate_append_constant
from data_utility.util.validate import (
    check_transcription,
    check_key,
    check_sample_rate,
    check_num_frames,
    check_wav_audio_tensor,
    check_gender,
    check_speaker_id,
)

########################################################################################
# general container of sample and ground truth values


@dataclass
class WavAudioDataSample:

    # identifier of sample
    key: str

    # tensor representing the (input) audio.
    # shape is [1, num_frames]
    audio_tensor: t.Tensor

    # sampling rate of the (original) audio
    sample_rate: int

    # the amount of frames this audio sample has
    audio_length_frames: int

    # speaker ID
    speaker_id: Optional[str]

    # transcription
    transcription: Optional[str]

    # gender
    gender: Optional[str]

    # language tag
    language_tag: Optional[str]

    def __len__(self):
        return self.audio_length_frames

    def __gt__(self, other):
        return len(self) > len(other)

    def __post_init__(self):
        assert check_key(self.key)
        assert check_num_frames(self.audio_length_frames)
        assert check_sample_rate(self.sample_rate)

        assert check_wav_audio_tensor(self.audio_tensor, self.audio_length_frames)

        if self.transcription is not None:
            assert check_transcription(self.transcription)

        if self.gender is not None:
            assert check_gender(self.gender)

        if self.speaker_id is not None:
            assert check_speaker_id(self.speaker_id, sample_id=self.key)


########################################################################################
# A batch of input for a speaker recognition network


@dataclass()
class SpeakerRecognitionBatch:
    # keys of each sample
    keys: List[str]

    # explicitly store batch size
    batch_size: int

    # audio tensors of each sample (in wav format)
    audio_tensor: t.Tensor  # shape [BATCH_SIZE, MAX_NUM_FRAMES]

    # store actual num of frames of each sample (padding excluded)
    audio_num_frames: List[int]

    # ground truth values of speaker identities as integer values
    # only required for training/validation
    id_tensor: Optional[t.Tensor] = None  # shape [BATCH_SIZE]

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.audio_tensor = self.audio_tensor.pin_memory()

        if self.id_tensor is not None:
            self.id_tensor = self.id_tensor.pin_memory()

        return self

    def __post_init__(self):
        assert self.batch_size == len(self.keys) == len(self.audio_num_frames)

        assert len(self.audio_tensor.shape) == 2

        assert self.audio_tensor.shape[0] == self.batch_size
        assert self.audio_tensor.shape[1] == max(self.audio_num_frames)

        if self.id_tensor is not None:
            assert len(self.id_tensor.shape) == 1
            assert self.id_tensor.shape[0] == self.batch_size

    @classmethod
    def from_sample_list(
        cls, samples: List[WavAudioDataSample], speaker_to_idx: Optional[Dict[str, int]]
    ):
        batch_size = len(samples)

        keys = []
        audio_tensors = []
        audio_num_frames = []
        speaker_id_idx = []

        for s in samples:
            keys.append(s.key)
            audio_tensors.append(s.audio_tensor)
            audio_num_frames.append(s.audio_length_frames)

            if speaker_to_idx is not None:
                speaker_id_idx.append(speaker_to_idx[s.speaker_id])

        audio_tensor = collate_append_constant(audio_tensors, variable_dim=1, value=0.0)
        audio_tensor = t.squeeze(audio_tensor, dim=1)

        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor[None, :]

        id_tensor = t.LongTensor(speaker_id_idx) if len(speaker_id_idx) > 0 else None

        return SpeakerRecognitionBatch(
            keys=keys,
            batch_size=batch_size,
            audio_tensor=audio_tensor,
            audio_num_frames=audio_num_frames,
            id_tensor=id_tensor,
        )


########################################################################################
# A batch of input for a gender recognition network


@dataclass()
class GenderRecognitionBatch:
    # keys of each sample
    keys: List[str]

    # explicitly store batch size
    batch_size: int

    # audio tensors of each sample (in wav format)
    audio_tensor: t.Tensor  # shape [BATCH_SIZE, MAX_NUM_FRAMES]

    # store actual num of frames of each sample (padding excluded)
    audio_num_frames: List[int]

    # ground truth values of gender identities as integer values
    id_tensor: t.Tensor  # shape [BATCH_SIZE]

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.audio_tensor = self.audio_tensor.pin_memory()
        self.id_tensor = self.id_tensor.pin_memory()

        return self

    def __post_init__(self):
        assert self.batch_size == len(self.keys) == len(self.audio_num_frames)

        assert len(self.audio_tensor.shape) == 2

        assert self.audio_tensor.shape[0] == self.batch_size
        assert self.audio_tensor.shape[1] == max(self.audio_num_frames)

        assert len(self.id_tensor.shape) == 1
        assert self.id_tensor.shape[0] == self.batch_size

    @classmethod
    def from_sample_list(
        cls, samples: List[WavAudioDataSample], gender_to_idx: Dict[str, int]
    ):
        batch_size = len(samples)

        keys = []
        audio_tensors = []
        audio_num_frames = []
        gender_id_idx = []

        for s in samples:
            keys.append(s.key)
            audio_tensors.append(s.audio_tensor)
            audio_num_frames.append(s.audio_length_frames)
            gender_id_idx.append(gender_to_idx[s.gender])

        audio_tensor = collate_append_constant(audio_tensors, variable_dim=1, value=0.0)
        audio_tensor = t.squeeze(audio_tensor, dim=1)

        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor[None, :]

        return GenderRecognitionBatch(
            keys=keys,
            batch_size=batch_size,
            audio_tensor=audio_tensor,
            audio_num_frames=audio_num_frames,
            id_tensor=t.LongTensor(gender_id_idx),
        )


########################################################################################
# A batch of input for a speech recognition network


@dataclass()
class SpeechRecognitionBatch:
    # keys of each sample
    keys: List[str]

    # explicitly store batch size
    batch_size: int

    # audio tensors of each sample (in wav format)
    audio_tensor: t.Tensor  # shape [BATCH_SIZE, MAX_NUM_FRAMES]

    # store actual num of frames of each sample (padding excluded)
    audio_num_frames: List[int]

    # ground truth values of letters in transcription as integer values
    transcriptions: List[str]

    # ground truth values of letters in transcription as integer values
    transcriptions_tensor: t.Tensor  # shape [BATCH_SIZE, MAX_LENGTH]

    # store actual number of letters in transcription (padding excluded)
    transcriptions_length: List[int]

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.audio_tensor = self.audio_tensor.pin_memory()
        self.transcriptions_tensor = self.transcriptions_tensor.pin_memory()
        return self

    def __post_init__(self):
        assert (
            self.batch_size
            == len(self.keys)
            == len(self.audio_tensor)
            == len(self.transcriptions_length)
        )

        assert (
            len(self.audio_tensor.shape) == len(self.transcriptions_tensor.shape) == 2
        )
        assert (
            self.audio_tensor.shape[0]
            == self.transcriptions_tensor.shape[0]
            == self.batch_size
        )

        assert max(self.audio_num_frames) == self.audio_tensor.shape[1]
        assert max(self.transcriptions_length) == self.transcriptions_tensor.shape[1]

    @classmethod
    def from_sample_list(
        cls, samples: List[WavAudioDataSample], char_to_idx: Dict[str, int]
    ):
        batch_size = len(samples)

        keys = []
        audio_tensors = []
        audio_num_frames = []
        transcription_strings = []
        transcription_tensors = []
        transcription_lengths = []

        for s in samples:
            keys.append(s.key)
            audio_tensors.append(s.audio_tensor)
            audio_num_frames.append(s.audio_length_frames)

            transcription_tensor = encode_transcription(s.transcription, char_to_idx)

            transcription_strings.append(s.transcription)
            transcription_tensors.append(transcription_tensor)
            transcription_lengths.append(len(s.transcription))

        audio_tensor = collate_append_constant(audio_tensors, variable_dim=1, value=0.0)
        audio_tensor = t.squeeze(audio_tensor, dim=1)

        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor[None, :]

        transcription_tensor = collate_append_constant(
            transcription_tensors, variable_dim=0
        )

        return SpeechRecognitionBatch(
            keys=keys,
            batch_size=batch_size,
            audio_tensor=audio_tensor,
            audio_num_frames=audio_num_frames,
            transcriptions=transcription_strings,
            transcriptions_tensor=transcription_tensor,
            transcriptions_length=transcription_lengths,
        )


########################################################################################
# a MTL batch for speech and speaker recognition


@dataclass()
class SpeechAndSpeakerRecognitionBatch:
    # keys of each sample
    keys: List[str]

    # explicitly store batch size
    batch_size: int

    # audio tensors of each sample (in wav format)
    audio_tensor: t.Tensor  # shape [BATCH_SIZE, MAX_NUM_FRAMES]

    # store actual num of frames of each sample (padding excluded)
    audio_num_frames: List[int]

    # ground truth values of letters in transcription as integer values
    transcriptions: List[str]

    # ground truth values of letters in transcription as integer values
    transcriptions_tensor: t.Tensor  # shape [BATCH_SIZE, MAX_LENGTH]

    # store actual number of letters in transcription (padding excluded)
    transcriptions_length: List[int]

    # ground truth values of speaker identities as integer values
    # only required for training/validation
    id_tensor: Optional[t.Tensor] = None  # shape [BATCH_SIZE]

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.audio_tensor = self.audio_tensor.pin_memory()
        self.transcriptions_tensor = self.transcriptions_tensor.pin_memory()

        if self.id_tensor is not None:
            self.id_tensor = self.id_tensor.pin_memory()

        return self

    def __post_init__(self):
        assert (
            self.batch_size
            == len(self.keys)
            == len(self.audio_tensor)
            == len(self.transcriptions_length)
        )

        assert (
            len(self.audio_tensor.shape) == len(self.transcriptions_tensor.shape) == 2
        )
        assert (
            self.audio_tensor.shape[0]
            == self.transcriptions_tensor.shape[0]
            == self.batch_size
        )

        assert max(self.audio_num_frames) == self.audio_tensor.shape[1]
        assert max(self.transcriptions_length) == self.transcriptions_tensor.shape[1]

        if self.id_tensor is not None:
            assert len(self.id_tensor.shape) == 1
            assert self.id_tensor.shape[0] == self.batch_size

    @classmethod
    def from_sample_list(
        cls,
        samples: List[WavAudioDataSample],
        char_to_idx: Dict[str, int],
        speaker_to_idx: Optional[Dict[str, int]],
    ) -> "SpeechAndSpeakerRecognitionBatch":
        batch_size = len(samples)

        keys = []
        audio_tensors = []
        audio_num_frames = []
        transcription_strings = []
        transcription_tensors = []
        transcription_lengths = []
        speaker_id_idx = []

        for s in samples:
            keys.append(s.key)
            audio_tensors.append(s.audio_tensor)
            audio_num_frames.append(s.audio_length_frames)

            if s.transcription is None:
                s.transcription = "none"  # hack for voxceleb transcriptions

            transcription_tensor = encode_transcription(s.transcription, char_to_idx)

            transcription_strings.append(s.transcription)
            transcription_tensors.append(transcription_tensor)
            transcription_lengths.append(len(s.transcription))

            if speaker_to_idx is not None:
                speaker_id_idx.append(speaker_to_idx[s.speaker_id])

        audio_tensor = collate_append_constant(audio_tensors, variable_dim=1, value=0.0)
        audio_tensor = t.squeeze(audio_tensor, dim=1)

        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor[None, :]

        transcription_tensor = collate_append_constant(
            transcription_tensors, variable_dim=0
        )

        id_tensor = t.LongTensor(speaker_id_idx) if len(speaker_id_idx) > 0 else None

        return SpeechAndSpeakerRecognitionBatch(
            keys=keys,
            batch_size=batch_size,
            audio_tensor=audio_tensor,
            audio_num_frames=audio_num_frames,
            transcriptions=transcription_strings,
            transcriptions_tensor=transcription_tensor,
            transcriptions_length=transcription_lengths,
            id_tensor=id_tensor,
        )
