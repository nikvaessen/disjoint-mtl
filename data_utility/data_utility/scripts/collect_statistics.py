#! /usr/bin/env python3
########################################################################################
#
# This script can be pointed to a directory of shards and collect statistics on
# the dataset contained within the shards.
#
# (Acts as an example of how to build a pipeline)
#
# Author(s): Nik Vaessen
########################################################################################

import pathlib

from collections import defaultdict
from typing import Tuple

import click
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utility.pipe.primitives.shard import load_audio_samples_from_shards
from data_utility.pipe.containers import WavAudioDataSample


########################################################################################
# entrypoint of script


class WavAudioDataSampleStatisticAggregator:
    def __init__(self, paths: Tuple[pathlib.Path]):
        self.paths = paths

        self.keys = []
        self.num_frames = []
        self.speaker_ids = set()
        self.transcription_lengths = []
        self.gender_map = dict()
        self.count_male_samples = 0
        self.count_female_samples = 0
        self.session_per_speaker = defaultdict(set)
        self.sessions = set()

    def __call__(self, x: WavAudioDataSample):
        assert isinstance(x, WavAudioDataSample)
        assert x.sample_rate == 16_000

        self.keys.append(x.key)
        self.num_frames.append(x.audio_length_frames)

        if x.speaker_id is not None:
            self.speaker_ids.add(x.speaker_id)

            session_id = x.key.split("/")[2]

            self.session_per_speaker[x.speaker_id].add(session_id)
            self.sessions.add(session_id)

        if x.transcription is not None:
            self.transcription_lengths.append(len(x.transcription))
        if x.gender is not None:
            if x.speaker_id not in self.gender_map:
                self.gender_map[x.speaker_id] = x.gender
            if x.gender == "m":
                self.count_male_samples += 1
            elif x.gender == "f":
                self.count_female_samples += 1
            else:
                raise ValueError(f"unknown gender {x.gender=}")

    def __repr__(self):
        representation = ""

        representation += "Aggregation of:\n"
        for p in self.paths:
            representation += f"\t{str(p)}\n"

        representation += "\n"
        representation += f"number of samples: {len(self.keys)}\n"
        representation += f"unique keys: {len(set(self.keys))}\n"

        representation += "\n"
        if len(self.speaker_ids) > 0:
            representation += f"number of speakers: {len(self.speaker_ids)}\n"
            if len(self.gender_map) > 0:
                representation += f"number of males: {sum(1 for v in self.gender_map.values() if v == 'm')}\n"
                representation += f"number of females: {sum(1 for v in self.gender_map.values() if v == 'f')}\n"
                representation += (
                    f"number of male audio samples: {self.count_male_samples}\n"
                )
                representation += (
                    f"number of female audio samples: {self.count_female_samples}\n"
                )
            else:
                representation += "no gender labels found"

            if len(self.speaker_ids) == len(self.session_per_speaker):
                number_of_sessions = [
                    len(v) for k, v in self.session_per_speaker.items()
                ]

                representation += f"\ntotal number of sessions: {len(self.sessions)}\n"
                representation += (
                    f"avg sessions per speaker: {np.mean(number_of_sessions):.2f}\n"
                )
                representation += (
                    f"min sessions per speaker: {np.min(number_of_sessions):.0f}\n"
                )
                representation += (
                    f"max sessions per speaker: {np.max(number_of_sessions):.0f}\n"
                )
            else:
                raise ValueError("inconsistent sessions")

        else:
            representation += "no speaker labels found\n"

        representation += "\n"
        sum_frames = sum(self.num_frames)
        min_frames = min(self.num_frames)
        max_frames = max(self.num_frames)
        avg_frames = sum_frames / len(self.keys)
        quantile_10 = np.quantile(self.num_frames, 0.1)
        quantile_20 = np.quantile(self.num_frames, 0.2)
        quantile_80 = np.quantile(self.num_frames, 0.8)
        quantile_90 = np.quantile(self.num_frames, 0.9)

        representation += f"avg number of frames: {avg_frames:.2f}\t({avg_frames / 16_000:.2f} seconds)\n"
        representation += (
            f"min number of frames: {min_frames}\t({min_frames / 16_000:.2f} seconds)\n"
        )
        representation += (
            f"max number of frames: {max_frames}\t({max_frames / 16_000:.2f} seconds)\n"
        )
        representation += (
            f"quantile 10%: {quantile_10:.2f}\t({quantile_10 / 16_000:.2f} seconds)\n"
        )
        representation += (
            f"quantile 20%: {quantile_20:.2f}\t({quantile_20 / 16_000:.2f} seconds)\n"
        )
        representation += (
            f"quantile 80%: {quantile_80:.2f}\t({quantile_80 / 16_000:.2f} seconds)\n"
        )
        representation += (
            f"quantile 90%: {quantile_90:.2f}\t({quantile_90 / 16_000:.2f} seconds)\n"
        )

        duration_seconds = sum_frames / 16_000
        representation += (
            f"\nduration: {duration_seconds:.2f} seconds"
            f"\t(~{duration_seconds / 3600:.2f} hours)\n"
        )

        representation += "\n"
        if len(self.transcription_lengths) > 0:
            min_tr_len = min(self.transcription_lengths)
            max_tr_len = max(self.transcription_lengths)
            avg_tr_len = sum(self.transcription_lengths) / len(self.keys)
            quantile_10 = np.quantile(self.transcription_lengths, 0.1)
            quantile_20 = np.quantile(self.transcription_lengths, 0.2)
            quantile_80 = np.quantile(self.transcription_lengths, 0.8)
            quantile_90 = np.quantile(self.transcription_lengths, 0.9)

            representation += (
                f"avg number of characters in transcription: {avg_tr_len:.2f}\n"
            )
            representation += (
                f"min number of characters in transcription: {min_tr_len}\n"
            )
            representation += (
                f"max number of characters in transcription: {max_tr_len}\n"
            )
            representation += (
                f"quantile 10%: {quantile_10:.2f} characters\n"
            )
            representation += (
                f"quantile 20%: {quantile_20:.2f} characters\n"
            )
            representation += (
                f"quantile 80%: {quantile_80:.2f} characters\n"
            )
            representation += (
                f"quantile 90%: {quantile_90:.2f} characters\n"
            )
        else:
            representation += "no transcription labels found"

        return representation


def aggregate(dirs: Tuple[pathlib.Path], allow_partial: bool, num_workers: int):
    dp = load_audio_samples_from_shards(list(dirs), allow_partial=allow_partial)

    ag = WavAudioDataSampleStatisticAggregator(dirs)
    for x in tqdm(DataLoader(dp, batch_size=None, num_workers=num_workers)):
        ag(x)

    return ag


@click.command()
@click.argument(
    "dirs",
    nargs=-1,
    type=pathlib.Path,
    required=True,
)
@click.option(
    "--partial",
    "allow_partial",
    type=bool,
    default=True,
    help="whether to read from partial shards",
)
@click.option(
    "--workers",
    "num_workers",
    type=int,
    default=1,
    help="number of workers in torch dataloader",
)
def main(dirs: Tuple[pathlib.Path], allow_partial: bool, num_workers: int):
    print(aggregate(dirs, allow_partial, num_workers))


if __name__ == "__main__":
    main()
