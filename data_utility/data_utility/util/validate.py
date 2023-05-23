########################################################################################
#
# Validate various inputs.
#
# Author(s): Nik Vaessen
########################################################################################

import pathlib
import re

from typing import Optional

import langcodes
import torch as t

########################################################################################
# validate methods

KNOWN_DATASET_ID = [
    "ls",  # librispeech
    "vc1",  # voxceleb1
    "vc2",  # voxceleb2
    "wsj",  # wall street journal
    "cv",  # common voice
    "cgn",  # corpus gesproken nederlands,
    "sw",  # switchboard
    "sre08",  # NIST 2008 speaker recognition evaluation challenge
    "hub5",  # eval set for switchboard
]


def check_key(key: str) -> bool:
    # must match pattern "<dataset_id>/<speaker_id>/<recording_id>/<utterance_id>
    match = re.fullmatch(r"[^\/]+\/[^\/]+\/[^\/]+\/[^\/]+$", key)

    if match is None:
        return False

    data_id, _, _, _ = key.split("/")

    return data_id in KNOWN_DATASET_ID


def check_speaker_id(speaker_id: str, sample_id: Optional[str] = None) -> bool:
    # must match pattern "<dataset_id>/<speaker_id>
    match = re.fullmatch(r"[^\/]+\/[^\/]+$", speaker_id)

    if sample_id is None:
        return match is not None

    if not check_key(sample_id):
        return False

    speaker_id_from_key = "/".join(sample_id.split("/")[0:2])
    return speaker_id_from_key == speaker_id


def check_recording_id(recording_id: str, sample_id: Optional[str] = None) -> bool:
    # must match pattern "<dataset_id>/<recording_id>
    match = re.fullmatch(r"[^\/]+\/[^\/]+$", recording_id)

    if sample_id is None:
        return match is not None

    if not check_key(sample_id):
        return False

    sample_id_split = sample_id.split("/")
    recording_id_from_key = "/".join([sample_id_split[0], sample_id[2]])

    return recording_id_from_key == recording_id


def check_audio_file_path(path: str) -> bool:
    return pathlib.Path(path).exists()


def check_wav_audio_tensor(audio_tensor: t.Tensor, expected_num_frames: int):
    return (
        isinstance(audio_tensor, t.Tensor)
        and isinstance(expected_num_frames, int)
        and len(audio_tensor.shape) == 2
        and audio_tensor.shape[0] == 1
        and audio_tensor.shape[1] == expected_num_frames
    )


def check_sample_rate(rate: int) -> bool:
    return isinstance(rate, int) and rate in [16_000]


def check_num_frames(num_frames: int):
    return isinstance(num_frames, int) and num_frames > 0


def check_gender(gender: str) -> bool:
    return gender in ["m", "f"]


def check_transcription(transcription: str):
    return isinstance(transcription, str) and len(transcription) >= 0


def check_language_tag(tag: str):
    return isinstance(tag, str) and langcodes.tag_is_valid(tag)
