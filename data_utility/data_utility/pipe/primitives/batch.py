########################################################################################
#
# Pipeline primitives for finalizing a batch before being given to a network.
#
# Author(s): Nik Vaessen
########################################################################################

from typing import Dict, List, Optional

from torchdata.datapipes.iter import IterDataPipe, Batcher, Mapper

from data_utility.pipe._patch import MaxTokenBucketizerIterDataPipe
from data_utility.pipe.containers import (
    WavAudioDataSample,
    SpeakerRecognitionBatch,
    SpeechRecognitionBatch,
    GenderRecognitionBatch,
    SpeechAndSpeakerRecognitionBatch,
)

#######################################################################################
# batches for speaker recognition


def map_to_speaker_batch(
    dp: IterDataPipe[WavAudioDataSample],
    speaker_id_to_idx: Optional[Dict[str, int]] = None,
    max_tokens: Optional[int] = None,
    max_len: Optional[int] = None,
    batch_size: Optional[int] = None,
    drop_last: Optional[bool] = None,  # must be given if batch_size is given
    buffer_size: int = 8,
):
    if max_tokens is None and batch_size is None:
        raise ValueError(f"one of {max_tokens=} or {batch_size=} must be given")
    if max_tokens is not None and batch_size is not None:
        raise ValueError(f"one of {max_tokens=} or {batch_size=} must be given")

    def map_fn(element: List[WavAudioDataSample]):
        assert isinstance(element, list)
        assert len(element) >= 1
        assert all([isinstance(x, WavAudioDataSample) for x in element])

        return SpeakerRecognitionBatch.from_sample_list(element, speaker_id_to_idx)

    if max_tokens is not None:
        dp = MaxTokenBucketizerIterDataPipe(
            dp,
            max_token_count=max_tokens,
            max_len=max_len,
            padded_tokens=True,
            buffer_size=buffer_size,
        )

        return Mapper(dp, fn=map_fn)
    elif batch_size is not None:
        if drop_last is None:
            raise ValueError(f"{drop_last=} must be defined if {batch_size=}")
        return Batcher(
            dp, batch_size=batch_size, drop_last=drop_last, wrapper_class=map_fn
        )


########################################################################################
# batches for gender recognition


def map_to_gender_batch(
    dp: IterDataPipe[WavAudioDataSample],
    batch_size: int,
    drop_last: bool,
    gender_to_idx: Dict[str, int],
):
    def map_fn(element: List[WavAudioDataSample]):
        assert isinstance(element, list)
        assert len(element) >= 1
        assert all([isinstance(x, WavAudioDataSample) for x in element])

        return GenderRecognitionBatch.from_sample_list(element, gender_to_idx)

    return Batcher(dp, batch_size=batch_size, drop_last=drop_last, wrapper_class=map_fn)


########################################################################################
# batches for speech recognition


def map_to_speech_batch(
    dp: IterDataPipe[WavAudioDataSample],
    char_to_idx: Dict[str, int],
    max_tokens: Optional[int] = None,
    max_len: Optional[int] = None,
    batch_size: Optional[int] = None,
    drop_last: Optional[bool] = None,  # must be given if batch_size is given
    buffer_size: int = 8,
):
    if max_tokens is None and batch_size is None:
        raise ValueError(f"one of {max_tokens=} or {batch_size=} must be given")
    if max_tokens is not None and batch_size is not None:
        raise ValueError(f"one of {max_tokens=} or {batch_size=} must be given")

    def map_fn(element: List[WavAudioDataSample]):
        assert isinstance(element, list)
        assert len(element) >= 1
        assert all([isinstance(x, WavAudioDataSample) for x in element])

        return SpeechRecognitionBatch.from_sample_list(element, char_to_idx)

    if max_tokens is not None:
        dp = MaxTokenBucketizerIterDataPipe(
            dp,
            max_token_count=max_tokens,
            max_len=max_len,
            padded_tokens=True,
            buffer_size=buffer_size,
        )

        return Mapper(dp, fn=map_fn)
    elif batch_size is not None:
        if drop_last is None:
            raise ValueError(f"{drop_last=} must be defined if {batch_size=}")
        return Batcher(
            dp, batch_size=batch_size, drop_last=drop_last, wrapper_class=map_fn
        )


########################################################################################
# batches for MTL speech and speaker


def map_to_speech_and_speaker_batch(
    dp: IterDataPipe[WavAudioDataSample],
    char_to_idx: Dict[str, int],
    speaker_id_to_idx: Optional[Dict[str, int]] = None,
    max_tokens: Optional[int] = None,
    batch_size: Optional[int] = None,
    drop_last: Optional[bool] = None,  # must be given if batch_size is given
    max_len: Optional[int] = None,
    buffer_size: int = 8,
):
    if max_tokens is None and batch_size is None:
        raise ValueError(f"one of {max_tokens=} or {batch_size=} must be given")
    if max_tokens is not None and batch_size is not None:
        raise ValueError(f"one of {max_tokens=} or {batch_size=} must be given")

    def map_fn(element: List[WavAudioDataSample]):
        assert isinstance(element, list)
        assert len(element) >= 1
        assert all([isinstance(x, WavAudioDataSample) for x in element])

        return SpeechAndSpeakerRecognitionBatch.from_sample_list(
            element, char_to_idx, speaker_id_to_idx
        )

    if max_tokens is not None:
        dp = MaxTokenBucketizerIterDataPipe(
            dp,
            max_token_count=max_tokens,
            max_len=max_len,
            padded_tokens=True,
            buffer_size=buffer_size,
        )

        return Mapper(dp, fn=map_fn)
    elif batch_size is not None:
        if drop_last is None:
            raise ValueError(f"{drop_last=} must be defined if {batch_size=}")
        return Batcher(
            dp, batch_size=batch_size, drop_last=drop_last, wrapper_class=map_fn
        )
