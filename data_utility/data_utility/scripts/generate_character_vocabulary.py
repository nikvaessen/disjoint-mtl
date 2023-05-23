#! /usr/bin/env python3
########################################################################################
#
# Script for generating a vocabulary file (character-based) on the transcriptions found
# in one or more shard folders.
#
# Author(s): Nik Vaessen
########################################################################################

import json
import pathlib

from typing import Tuple

import click

from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utility.eval.speech.transform import get_default_token_list
from data_utility.pipe.primitives.shard import load_audio_samples_from_shards
from data_utility.pipe.containers import WavAudioDataSample


########################################################################################
# functionality to collect all letters in all transcripts


class WavAudioDataSampleVocabularyAggregator:
    def __init__(self):
        self.vocab = set()

    def __call__(self, x: WavAudioDataSample):
        assert isinstance(x, WavAudioDataSample)

        if x.transcription is not None:
            for c in x.transcription:
                self.vocab.add(c)
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

    # ensure blank is always index 0 for CTC loss, and space is already defined as |
    list_chars = get_default_token_list() + sorted([c for c in ag.vocab if c != " "])
    char_to_idx = {c: i for i, c in enumerate(list_chars)}
    idx_to_char = {v: k for k, v in char_to_idx.items()}

    vocab_dict = {
        "characters": list_chars,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
    }

    with json_path.open("w") as f:
        json.dump(vocab_dict, f)


if __name__ == "__main__":
    main()
