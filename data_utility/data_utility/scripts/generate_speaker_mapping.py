#! /usr/bin/env python3
########################################################################################
#
# Script for generating a file which maps each speaker identity to a particular index
# for classification purposes. The file also provides a map of speaker_id to gender.
#
# Author(s): Nik Vaessen
########################################################################################

import json
import pathlib

from typing import Tuple

import click

from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utility.pipe.primitives.shard import load_audio_samples_from_shards
from data_utility.util.various import sort_speaker_id_key
from data_utility.scripts.generate_speaker_trials import (
    WavAudioDataSampleTrialAggregator,
)


########################################################################################
# entrypoint of script


@click.command()
@click.argument(
    "dirs",
    nargs=-1,
    type=pathlib.Path,
    required=True,
)
@click.option(
    "--out",
    "json_path",
    type=pathlib.Path,
    required=True,
)
@click.option(
    "--workers",
    "num_workers",
    type=int,
    default=1,
    help="number of workers in torch dataloader",
)
def main(dirs: Tuple[pathlib.Path], json_path: pathlib.Path, num_workers: int):
    dp = load_audio_samples_from_shards(list(dirs), allow_partial=True)

    ag = WavAudioDataSampleTrialAggregator()
    for x in tqdm(DataLoader(dp, batch_size=None, num_workers=num_workers)):
        ag(x)

    # ensure blank is always index 0 for CTC loss
    speaker_ids = sorted(list(ag.speaker_ids), key=sort_speaker_id_key)
    speaker_to_idx = {c: i for i, c in enumerate(speaker_ids)}
    idx_to_speaker = {v: k for k, v in speaker_to_idx.items()}

    speaker_mapping = {
        "speakers": speaker_ids,
        "speaker_to_idx": speaker_to_idx,
        "idx_to_speaker": idx_to_speaker,
        "gender": ag.get_gender_mapping(),
    }

    with json_path.open("w") as f:
        json.dump(speaker_mapping, f)


if __name__ == "__main__":
    main()
