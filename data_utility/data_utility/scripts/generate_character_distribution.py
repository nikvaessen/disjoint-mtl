#! /usr/bin/env python3
########################################################################################
#
# Script for generating a distribution over the frequency of all characters
# in the transcriptions of one or more shard folders.
#
# Author(s): Nik Vaessen
########################################################################################

import json
import pathlib

from collections import defaultdict
from typing import Tuple

import click

from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utility.eval.speech.transform import (
    get_default_token_list,
    WORD_DELIM_TOKEN,
)
from data_utility.pipe.primitives.shard import load_audio_samples_from_shards
from data_utility.pipe.containers import WavAudioDataSample


########################################################################################
# functionality to collect all letters in all transcripts


class WavAudioDataSampleVocabularyAggregator:
    def __init__(self):
        self.vocab = defaultdict(int)

    def __call__(self, x: WavAudioDataSample):
        assert isinstance(x, WavAudioDataSample)

        if x.transcription is not None:
            for c in x.transcription:
                self.vocab[c] += 1
        else:
            raise ValueError("sample is missing transcription")


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
def main(dirs: Tuple[pathlib.Path], json_path: pathlib.Path):
    dp = load_audio_samples_from_shards(list(dirs), allow_partial=True)

    ag = WavAudioDataSampleVocabularyAggregator()
    for x in tqdm(DataLoader(dp, batch_size=None, num_workers=0)):
        ag(x)

    # compute distribution
    vocab = ag.vocab
    total_count = sum(v for v in vocab.values())

    distribution = {k: 0 for k in get_default_token_list()} | {
        k: v / total_count
        for k, v in sorted(vocab.items(), key=lambda tup: tup[0])
        if k != " "
    }
    distribution[WORD_DELIM_TOKEN] = vocab[" "] / total_count

    # save collected data to disk
    with json_path.open("w") as f:
        json.dump(distribution, f)


if __name__ == "__main__":
    main()
