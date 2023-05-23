########################################################################################
#
# Transcribe a dataset with whisper's ASR model
#
# Author(s): Nik Vaessen
########################################################################################

import json
import pathlib

from typing import Tuple, Optional

import click
import torch.cuda

from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utility.pipe.primitives.shard import (
    load_audio_samples_from_shards,
    find_shards,
)
from data_utility.pipe.containers import WavAudioDataSample

import whisper

########################################################################################
# functionality to collect all letters in all transcripts


class WavAudioDataSampleTranscriptionAggregator:
    def __init__(
        self,
        model: str,
        out: pathlib.Path,
        save_interval: int = 50,
        cache: Optional[pathlib.Path] = None,
    ):
        self.model = whisper.load_model(model)

        self.out_path = out
        self.cache_path = cache

        if torch.cuda.is_available():
            self.model.to("cuda")

        self.cache_dict = self.load_cache()
        self.results_dict = {}

        self.save_interval = save_interval
        self.counter = 0

    def __call__(self, x: WavAudioDataSample):
        assert isinstance(x, WavAudioDataSample)

        if torch.cuda.is_available():
            pass

        key = x.key

        if key in self.cache_dict:
            self.results_dict[key] = self.cache_dict[key]
        else:
            result = self.model.transcribe(x.audio_tensor.squeeze(), fp16=True)
            self.results_dict[key] = result

        self.counter += 1
        if self.counter > self.save_interval:
            self.save()
            self.counter = 0

    def load_cache(self):
        if self.cache_path is not None and self.cache_path.exists():
            with self.cache_path.open("r") as f:
                return json.load(f)
        else:
            return {}

    def save(self):
        with self.out_path.open("w") as f:
            json.dump(self.results_dict, f)


########################################################################################
# entrypoint of script


@click.command()
@click.argument(
    "dirs",
    nargs=-1,
    type=pathlib.Path,
    required=True,
)
@click.option("--model", type=str, required=True, help="which whisper model to use")
@click.option(
    "--cache",
    "cache_file",
    type=pathlib.Path,
    default=None,
    help="instead of doing inference, read results from cache if key of sample is included",
)
def main(
    dirs: Tuple[pathlib.Path], model: str, cache_file: Optional[pathlib.Path] = None
):
    shard_list = [pathlib.Path(p) for p in find_shards([*dirs], "*.*.tar*", True)]

    if len(shard_list) == 0:
        print(f"no shards found at {dirs=}")

    for shard in shard_list:
        out_file = shard.parent / f"{shard.stem}.whisper.{model}.transcript.json"

        print(f"writing transcripts to {out_file}")

        ag = WavAudioDataSampleTranscriptionAggregator(
            model, out=out_file, cache=cache_file
        )
        dp = load_audio_samples_from_shards(pathlib.Path(shard), allow_partial=True)

        # compute the transcriptions
        for x in tqdm(DataLoader(dp, batch_size=None, num_workers=0)):
            ag(x)
        ag.save()


if __name__ == "__main__":
    main()
