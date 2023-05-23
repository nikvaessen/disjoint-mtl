########################################################################################
#
# Pipeline building block for loading data from shards (as .tar files)
#
# Author(s): Nik Vaessen
########################################################################################

import pathlib
import json
import fnmatch

from typing import List, Union, Tuple, Dict, Optional

import torchaudio

from torch.utils.data.datapipes.utils.common import StreamWrapper
from torchdata.datapipes.iter import (
    FileLister,
    FileOpener,
    TarArchiveLoader,
    Mapper,
    WebDataset,
    Shuffler,
    ShardingFilter,
    Header,
    Filter,
    IterDataPipe,
)

from data_utility.pipe.containers import WavAudioDataSample
from data_utility.util.various import search_sample_key


########################################################################################
# utility functions for initial loading of elements in a pipeline


def decode_wav(value: StreamWrapper):
    assert isinstance(value, StreamWrapper)

    value, sample_rate = torchaudio.load(value)
    assert sample_rate == 16_000

    return value


def decode_json(value: StreamWrapper):
    assert isinstance(value, StreamWrapper)

    return json.load(value)


def decode(element: Tuple[str, StreamWrapper]):
    assert isinstance(element, tuple) and len(element) == 2
    key, value = element

    assert isinstance(key, str)
    assert isinstance(value, StreamWrapper)

    if key.endswith(".wav"):
        value = decode_wav(value)

    if key.endswith(".json"):
        value = decode_json(value)

    return key, value


def construct_data_sample(element: Dict):
    assert isinstance(element, dict)

    json = element[".json"]
    wav = element[".wav"]

    return WavAudioDataSample(
        key=json["sample_id"],
        audio_tensor=wav,
        sample_rate=json["sample_rate"],
        audio_length_frames=json["num_frames"],
        speaker_id=json["speaker_id"],
        transcription=json["transcription"],
        gender=json["gender"],
        language_tag=json["language_tag"],
    )


def wrap_filter_on_speaker_id(filtered_id: List[str]):
    membership_set = set(filtered_id)

    def fn(element):
        key_path, _ = element

        key = search_sample_key(key_path)

        if key is None:
            raise ValueError(f"unable to extract key from {key_path}")

        speaker_id_from_key = "/".join(key.split("/")[0:2])

        return speaker_id_from_key in membership_set

    return fn


########################################################################################
# generic loading from archives


def find_shards(
    dirs_or_files: Union[pathlib.Path, List[pathlib.Path]],
    shard_file_pattern: str,
    allow_partial: bool,
) -> List[str]:
    if isinstance(dirs_or_files, pathlib.Path):
        dirs_or_files = [dirs_or_files]

    tar_files = []

    for dir_or_file in dirs_or_files:
        if dir_or_file.is_dir():
            tar_files.extend(dir_or_file.rglob(shard_file_pattern))
        if dir_or_file.is_file():
            if fnmatch.fnmatch(dir_or_file.name, shard_file_pattern):
                tar_files.append(dir_or_file)

    # remove potential duplicates by casting to set
    tar_files = list(
        set([str(f) for f in tar_files if allow_partial or ".partial" not in f.name])
    )

    return tar_files


def load_audio_samples_from_shards(
    dirs_or_files: Union[pathlib.Path, List[pathlib.Path]],
    shard_file_pattern: str = "*.*.tar*",
    tar_read_mode: str = "r",  # see https://docs.python.org/3/library/tarfile.html#tarfile.open
    allow_partial: bool = False,
    shuffle_buffer: int = 100,
    num_equal_workers: int = None,
    filter_speaker_ids: Optional[List[str]] = None,
) -> IterDataPipe[WavAudioDataSample]:
    shard_list = find_shards(dirs_or_files, shard_file_pattern, allow_partial)

    print(
        "creating a data pipeline from the following tar files:\n",
        *[f"\t{s}\n" for s in sorted(shard_list)],
        end="",
        flush=True,
    )

    if len(shard_list) <= 0:
        if isinstance(dirs_or_files, list):
            raise ValueError(
                f"unable to find shards in {[str(t) for t in dirs_or_files]}"
            )
        else:
            raise ValueError(f"unable to find shards in {dirs_or_files}")

    # stream of strings representing each shard
    dp = FileLister(shard_list)

    # shuffle the stream so order of shards in epoch differs
    dp = Shuffler(dp, buffer_size=len(shard_list))

    # optionally, make sure each worker receives the same number of shards
    if num_equal_workers:
        dp = Header(dp, limit=len(shard_list) // num_equal_workers * num_equal_workers)

    # each worker only sees 1/n elements
    dp = ShardingFilter(dp)

    # map strings of paths to file handles
    dp = FileOpener(dp, mode="b")

    # expand each file handle to a stream of all files in the tar
    dp = TarArchiveLoader(dp, mode=tar_read_mode)

    # optionally, filter out speaker IDs before decoding
    if filter_speaker_ids is not None:
        dp = Filter(dp, wrap_filter_on_speaker_id(filter_speaker_ids))

    # decode each file in the tar to the expected python dataformat
    dp = Mapper(dp, decode)

    # each file in the tar is expected to have the format `{key}.{ext}
    # this groups all files with the same key into one dictionary
    dp = WebDataset(dp)

    # map the dictionary of files with same key to `WavAudioDataSample` dataclass
    dp = Mapper(dp, construct_data_sample)

    # create a buffer so that batches vary across epochs
    dp = Shuffler(dp, buffer_size=shuffle_buffer)

    return dp
