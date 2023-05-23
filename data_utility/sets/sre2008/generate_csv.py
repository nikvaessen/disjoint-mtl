########################################################################################
#
# This script can be used to generate the CSV file(s) which will be used to write
# the sre 2008 eval dataset into shards.
#
# Author(s): Nik Vaessen
########################################################################################

import pathlib

from typing import List

import click
import torchaudio
import yaspin

from tqdm import tqdm
from data_utility.util.csv import ShardCsvSample


########################################################################################
# entrypoint of script


@click.command()
@click.option(
    "--dir",
    "dir_path",
    type=pathlib.Path,
    required=True,
    help="path the root directory of sre08 data in wav format",
)
@click.option(
    "--trial",
    "trial_file",
    type=pathlib.Path,
    required=True,
    help="path to trial file, only files present in file will be included",
)
@click.option(
    "--csv",
    "csv_file",
    type=pathlib.Path,
    required=True,
    help="path to write output csv file to",
)
def main(
    dir_path: pathlib.Path,
    csv_file: pathlib.Path,
    trial_file: pathlib.Path,
):
    samples: List[ShardCsvSample] = []
    files = [f.absolute() for f in dir_path.rglob("*.wav") if f.is_file()]

    with trial_file.open("r") as f:
        lines = [f.strip().split(" ") for f in f.readlines() if len(f.strip()) > 0]
        keys = set()

        for ln in lines:
            a, b, gt = ln
            keys.add(a)
            keys.add(b)

    for audio_file in tqdm(files):
        if audio_file.stem not in keys:
            continue

        meta = torchaudio.info(str(audio_file))

        samples.append(
            ShardCsvSample(
                key=f"sre08/unk/unk/{audio_file.stem}",
                path=str(audio_file),
                num_frames=meta.num_frames,
                sample_rate=meta.sample_rate,
                speaker_id="sre08/unk",
                recording_id="sre08/unk",
            )
        )

    with yaspin.yaspin(text=f"writing {csv_file}"):
        csv_file.parent.mkdir(exist_ok=True, parents=True)
        ShardCsvSample.to_csv(csv_file, samples)


if __name__ == "__main__":
    main()
