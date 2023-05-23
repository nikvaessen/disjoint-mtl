########################################################################################
#
# Build a data pipe based on a configuration.
#
# Author(s): Nik Vaessen
########################################################################################
import functools
import pathlib

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Union, Dict, Optional

from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterDataPipe, Shuffler, Filter

from data_utility.pipe.containers import WavAudioDataSample
from data_utility.pipe.primitives.shard import load_audio_samples_from_shards
from data_utility.pipe.primitives.chunk import ChunkAudioSampleIterDataPipe
from data_utility.pipe.primitives.batch import (
    map_to_speaker_batch,
    map_to_speech_batch,
    map_to_gender_batch,
    map_to_speech_and_speaker_batch,
)


########################################################################################
# abstract API


class DataPipeBuilder:
    @abstractmethod
    def get_pipe(
        self,
        shard_dirs: Union[pathlib.Path, List[pathlib.Path]],
        shard_file_pattern: str,
    ) -> IterDataPipe:
        pass

    @abstractmethod
    def wrap_pipe(self, dp: IterDataPipe) -> DataLoader:
        pass


########################################################################################
# builder for a speech recognition data pipe


@dataclass
class SpeechRecognitionDataPipeBuilderConfig:
    # potential compression of shards
    tar_read_mode: str  # depends on compression or not

    # parameters determining randomness
    shuffle_buffer_before: int
    shuffle_buffer_after: int
    bucket_buffer: int

    # filter on language_tags
    language_tags: List["str"]

    # logic for giving each worker equal number of data
    allow_partial_shards: bool
    num_workers: int
    drop_last: Optional[bool] = None  # must be defined is batch_size is defined

    # batching (one must be defined)
    max_tokens: Optional[int] = None
    batch_size: Optional[int] = None
    max_audio_frames: Optional[int] = None  # maximum length of a possible sample
    max_transcription_frames: Optional[int] = None  # maximum length of gt


def filter_samples(
    x: WavAudioDataSample,
    allowed_language_tags: Optional[List[str]] = None,
    max_audio_frames: Optional[int] = None,
    max_gt_frames: Optional[int] = None,
    allow_empty_transcription: bool = False,
):
    has_non_empty_transcription = allow_empty_transcription or (
        x.transcription is not None and len(x.transcription) > 0
    )
    has_valid_tag = allowed_language_tags is None or (
        x.language_tag is not None and x.language_tag in allowed_language_tags
    )
    valid_audio_frames = (
        max_audio_frames is None or x.audio_length_frames < max_audio_frames
    )
    valid_gt_frames = (
        x.transcription is None
        or max_gt_frames is None
        or len(x.transcription) < max_gt_frames
    )

    return (
        has_non_empty_transcription
        and has_valid_tag
        and valid_audio_frames
        and valid_gt_frames
    )


class SpeechRecognitionDataPipeBuilder(DataPipeBuilder):
    def __init__(self, cfg: SpeechRecognitionDataPipeBuilderConfig):
        self.cfg = cfg
        self.char_to_idx = None

    def set_char_to_idx(self, character_vocabulary: Dict[str, int]):
        self.char_to_idx = character_vocabulary

    def get_pipe(
        self,
        shard_dirs: Union[pathlib.Path, List[pathlib.Path]],
        shard_file_pattern: str,
    ):
        if self.char_to_idx is None:
            raise ValueError("set char_to_idx first")

        # First get a stream of WavAudioSamples
        dp = load_audio_samples_from_shards(
            dirs_or_files=shard_dirs,
            shard_file_pattern=shard_file_pattern,
            tar_read_mode=self.cfg.tar_read_mode,
            allow_partial=self.cfg.allow_partial_shards,
            shuffle_buffer=self.cfg.shuffle_buffer_before,
            num_equal_workers=self.cfg.num_workers,
        )

        # filter on non-empty transcription and english language
        dp = Filter(
            dp,
            functools.partial(
                filter_samples,
                allowed_language_tags=self.cfg.language_tags,
                max_audio_frames=self.cfg.max_audio_frames,
                max_gt_frames=self.cfg.max_transcription_frames,
            ),
        )

        # convert to batches
        dp = map_to_speech_batch(
            dp,
            char_to_idx=self.char_to_idx,
            max_tokens=self.cfg.max_tokens,
            batch_size=self.cfg.batch_size,
            drop_last=self.cfg.drop_last,
            max_len=self.cfg.max_audio_frames,
            buffer_size=self.cfg.bucket_buffer,
        )

        # shuffle again
        dp = Shuffler(dp, buffer_size=self.cfg.shuffle_buffer_after)

        return dp

    def wrap_pipe(self, dp: IterDataPipe) -> DataLoader:
        return DataLoader(
            dp,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )


########################################################################################
# builder for a speaker recognition data pipe


@dataclass
class SpeakerRecognitionDataPipeBuilderConfig:
    # potential compression of shards
    tar_read_mode: str  # depends on compression or not

    # parameters determining randomness
    shuffle_buffer_before: int
    shuffle_buffer_after: int
    bucket_buffer: int

    # logic related to chunking audio
    # before batches are created
    chunk_strategy: str
    chunk_size_sec: float

    # logic for giving each worker equal number of data
    allow_partial_shards: bool
    num_workers: int
    drop_last: Optional[bool] = None  # must be defined is batch_size is defined

    # batching (one must be defined)
    max_tokens: Optional[int] = None
    batch_size: Optional[int] = None
    max_audio_frames: Optional[int] = None  # maximum length of a possible sample


class SpeakerRecognitionDataPipeBuilder(DataPipeBuilder):
    def __init__(self, cfg: SpeakerRecognitionDataPipeBuilderConfig):
        self.cfg = cfg
        self.speaker_to_idx = None

    def set_speaker_to_idx(self, speaker_to_idx: Dict[str, int]):
        self.speaker_to_idx = speaker_to_idx

    def get_pipe(
        self,
        shard_dirs: Union[pathlib.Path, List[pathlib.Path]],
        shard_file_pattern: str,
    ):
        # First get a stream of WavAudioSamples
        dp = load_audio_samples_from_shards(
            dirs_or_files=shard_dirs,
            shard_file_pattern=shard_file_pattern,
            tar_read_mode=self.cfg.tar_read_mode,
            allow_partial=self.cfg.allow_partial_shards,
            shuffle_buffer=self.cfg.shuffle_buffer_before,
            num_equal_workers=self.cfg.num_workers,
        )

        # chunk audio to a particular size
        dp = ChunkAudioSampleIterDataPipe(
            dp,
            chunk_strategy=self.cfg.chunk_strategy,
            chunk_size_seconds=self.cfg.chunk_size_sec,
            sample_rate=16_000,
        )

        # convert to batches
        dp = map_to_speaker_batch(
            dp,
            speaker_id_to_idx=self.speaker_to_idx,
            max_tokens=self.cfg.max_tokens,
            batch_size=self.cfg.batch_size,
            drop_last=self.cfg.drop_last,
            max_len=self.cfg.max_audio_frames,
            buffer_size=self.cfg.bucket_buffer,
        )

        # shuffle again
        dp = Shuffler(dp, buffer_size=self.cfg.shuffle_buffer_after)

        return dp

    def wrap_pipe(self, dp: IterDataPipe) -> DataLoader:
        return DataLoader(
            dp,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )


########################################################################################
# builder for a speaker recognition data pipe


@dataclass
class GenderRecognitionDataPipeBuilderConfig:
    # potential compression of shards
    tar_read_mode: str  # depends on compression or not

    # parameters determining randomness
    shuffle_buffer: int

    # chunk
    chunk_strategy: str
    chunk_size_sec: float

    # batching
    batch_size: int

    # logic for giving each worker equal number of data
    allow_partial_shards: bool
    num_workers: int
    drop_last: bool


class GenderRecognitionDataPipeBuilder(DataPipeBuilder):
    def __init__(self, cfg: GenderRecognitionDataPipeBuilderConfig):
        self.cfg = cfg
        self.gender_to_idx = None

    def set_gender_idx(
        self, female_str: str, female_idx: int, male_str: str, male_idx: int
    ):
        assert male_idx != female_idx
        assert isinstance(male_idx, int)
        assert isinstance(female_idx, int)
        assert male_idx + female_idx == 1

        self.gender_to_idx = {
            male_str: male_idx,
            female_str: female_idx,
        }

    def get_pipe(
        self,
        shard_dirs: Union[pathlib.Path, List[pathlib.Path]],
        shard_file_pattern: str,
    ):
        if self.gender_to_idx is None:
            raise ValueError("set gender_to_idx")

        # First get a stream of WavAudioSamples
        dp = load_audio_samples_from_shards(
            dirs_or_files=shard_dirs,
            shard_file_pattern=shard_file_pattern,
            tar_read_mode=self.cfg.tar_read_mode,
            allow_partial=self.cfg.allow_partial_shards,
            shuffle_buffer=self.cfg.shuffle_buffer,
            num_equal_workers=self.cfg.num_workers,
        )

        # chunk audio to a particular size
        dp = ChunkAudioSampleIterDataPipe(
            dp,
            chunk_strategy=self.cfg.chunk_strategy,
            chunk_size_seconds=self.cfg.chunk_size_sec,
            sample_rate=16_000,
        )

        # convert to batches
        dp = map_to_gender_batch(
            dp,
            batch_size=self.cfg.batch_size,
            drop_last=self.cfg.drop_last,
            gender_to_idx=self.gender_to_idx,
        )

        return dp

    def wrap_pipe(self, dp: IterDataPipe) -> DataLoader:
        return DataLoader(
            dp,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )


########################################################################################
# a multi-task batch for speech and speaker recognition


@dataclass
class SpeechAndSpeakerRecognitionDataPipeBuilderConfig:
    # potential compression of shards
    tar_read_mode: str  # depends on compression or not

    # parameters determining randomness
    shuffle_buffer_before: int
    shuffle_buffer_after: int
    bucket_buffer: int

    # filter on language_tags
    language_tags: List["str"]

    # logic for giving each worker equal number of data
    allow_partial_shards: bool
    num_workers: int
    drop_last: Optional[bool] = None  # must be defined is batch_size is defined

    # batching (one must be defined)
    max_tokens: Optional[int] = None
    batch_size: Optional[int] = None
    max_audio_frames: Optional[int] = None  # maximum length of a possible sample
    max_transcription_frames: Optional[int] = None  # maximum length of gt


class SpeechAndSpeakerRecognitionDataPipeBuilder(DataPipeBuilder):
    def __init__(
        self,
        cfg: SpeechAndSpeakerRecognitionDataPipeBuilderConfig,
    ):
        self.cfg = cfg
        self.speaker_to_idx = None
        self.char_to_idx = None

    def set_speaker_to_idx(self, speaker_to_idx: Dict[str, int]):
        self.speaker_to_idx = speaker_to_idx

    def set_char_to_idx(self, character_vocabulary: Dict[str, int]):
        self.char_to_idx = character_vocabulary

    def get_pipe(
        self,
        shard_dirs: Union[pathlib.Path, List[pathlib.Path]],
        shard_file_pattern: str,
        ignore_language_tag_filter: bool = False,
    ):
        # First get a stream of WavAudioSamples
        dp = load_audio_samples_from_shards(
            dirs_or_files=shard_dirs,
            shard_file_pattern=shard_file_pattern,
            tar_read_mode=self.cfg.tar_read_mode,
            allow_partial=self.cfg.allow_partial_shards,
            shuffle_buffer=self.cfg.shuffle_buffer_before,
            num_equal_workers=self.cfg.num_workers,
        )

        # filter on non-empty transcription and english language
        dp = Filter(
            dp,
            functools.partial(
                filter_samples,
                allowed_language_tags=None
                if ignore_language_tag_filter
                else self.cfg.language_tags,
                max_audio_frames=self.cfg.max_audio_frames,
                max_gt_frames=self.cfg.max_transcription_frames,
                allow_empty_transcription=ignore_language_tag_filter
            ),
        )

        # convert to batches
        dp = map_to_speech_and_speaker_batch(
            dp,
            char_to_idx=self.char_to_idx,
            max_tokens=self.cfg.max_tokens,
            batch_size=self.cfg.batch_size,
            drop_last=self.cfg.drop_last,
            max_len=self.cfg.max_audio_frames,
            buffer_size=self.cfg.bucket_buffer,
            speaker_id_to_idx=self.speaker_to_idx,
        )

        # shuffle again
        dp = Shuffler(dp, buffer_size=self.cfg.shuffle_buffer_after)

        return dp

    def wrap_pipe(self, dp: IterDataPipe) -> DataLoader:
        return DataLoader(
            dp,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )
