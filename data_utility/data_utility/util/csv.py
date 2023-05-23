########################################################################################
#
# Define a CSV format for describing a dataset for efficient manipulation without
# accessing the underlying data.
#
# Author(s): Nik Vaessen
########################################################################################

import pathlib
import re

from dataclasses import dataclass
from typing import List, Union

import pandas as pd
import pandera as pa
import langcodes

from data_utility.util.validate import (
    check_speaker_id,
    check_key,
    check_audio_file_path,
    check_num_frames,
    check_sample_rate,
    check_gender,
    check_transcription,
    check_recording_id,
    check_language_tag,
)

########################################################################################
# helpers for input CSV file


audio_data_schema = pa.DataFrameSchema(
    {
        "key": pa.Column(
            str, checks=pa.Check(check_key, element_wise=True), unique=True, coerce=True
        ),
        "path": pa.Column(
            str, checks=pa.Check(check_audio_file_path, element_wise=True), unique=True
        ),
        "num_frames": pa.Column(
            int, checks=pa.Check(check_num_frames, element_wise=True), coerce=True
        ),
        "sample_rate": pa.Column(
            int, pa.Check(check_sample_rate, element_wise=True), coerce=True
        ),
        "speaker_id": pa.Column(
            str,
            checks=pa.Check(check_speaker_id, element_wise=True),
            nullable=True,
        ),
        "recording_id": pa.Column(
            str,
            checks=pa.Check(check_recording_id, element_wise=True),
            nullable=True,
        ),
        "gender": pa.Column(
            str, pa.Check(check_gender, element_wise=True), nullable=True
        ),
        "language_tag": pa.Column(
            str, checks=pa.Check(check_language_tag, element_wise=True), nullable=True
        ),
        "transcription": pa.Column(
            str, checks=pa.Check(check_transcription, element_wise=True), nullable=True
        ),
    },
    strict=True,
    coerce=True,
    ordered=True,
)


@dataclass()
class ShardCsvSample:
    key: str
    path: str
    num_frames: int
    sample_rate: int
    speaker_id: Union[str, None] = None
    recording_id: Union[str, None] = None
    gender: Union[str, None] = None
    language_tag: Union[str, None] = None
    transcription: Union[str, None] = None

    def __post_init__(self):
        assert check_key(self.key)
        assert check_audio_file_path(self.path)
        assert check_num_frames(self.num_frames)
        assert check_sample_rate(self.sample_rate)

        if self.transcription is not None:
            # remove multiple spaces, tabs and newlines otherwise csv's are broken
            self.transcription = re.sub("\s+", " ", self.transcription)
            self.transcription = self.transcription.strip()
            assert check_transcription(self.transcription)

        if self.gender is not None:
            assert check_gender(self.gender)

        if self.speaker_id is not None:
            assert check_speaker_id(self.speaker_id, sample_id=self.key)

        if self.language_tag is not None:
            self.language_tag = langcodes.standardize_tag(self.language_tag, macro=True)

    @classmethod
    def to_dataframe(cls, samples: List["ShardCsvSample"]):
        df = pd.DataFrame([s.__dict__ for s in samples])
        audio_data_schema(df)

        return df

    @classmethod
    def to_csv(
        cls, path: pathlib.Path, samples: Union[List["ShardCsvSample"], pd.DataFrame]
    ):
        if isinstance(samples, list):
            df = cls.to_dataframe(samples)
        elif isinstance(samples, pd.DataFrame):
            df = samples
            audio_data_schema(df)
        else:
            raise ValueError(f"unvalid type {type(samples)=}")

        df = df.sort_values("key", ascending=True, ignore_index=True)
        df.to_csv(str(path), index=False, sep="\t", quotechar='"', escapechar="\\")

    @classmethod
    def from_csv(cls, path: pathlib.Path) -> pd.DataFrame:
        df = pd.read_csv(str(path), sep="\t", quotechar='"', escapechar="\\")
        audio_data_schema(df)

        return df
