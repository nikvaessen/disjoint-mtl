########################################################################################
#
# This script can be used to generate the CSV file(s) which will be used to write
# the LibriSpeech dataset into shards.
#
# Author(s): Nik Vaessen
########################################################################################

import pathlib
from typing import List

import click
import pandas as pd
import torchaudio
import yaspin

from tqdm import tqdm

from data_utility.util.csv import ShardCsvSample


########################################################################################
# logic for traversing dataset folder and writing info to CSV file


def load_speaker_meta(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        skiprows=12,
        header=None,
        names=["id", "sex", "subset", "minutes", "name"],
        sep=r"\s+\|\s+",
        engine="python",
    )

    return df


def traverse_split(
    path: pathlib.Path, df_speaker_meta: pd.DataFrame, extension: str
) -> List[ShardCsvSample]:
    all_samples = []

    # we find each '*-*.trans.txt` file, and manually determine the path to
    # the audio file for each transcription
    with yaspin.yaspin(text=f"recursively globbing {path}"):
        files = [f for f in path.rglob("*-*.trans.txt")]

    for trans_file in tqdm(files):
        parent_folder = trans_file.parent

        # load transcriptions
        with trans_file.open("r") as f:
            lines = f.readlines()

        # split transcriptions into ID and sentence
        samples_in_file = [line.strip().lower().split(" ") for line in lines]
        samples_in_file = [(line[0], " ".join(line[1:])) for line in samples_in_file]

        for sample_id, transcription in samples_in_file:
            path = parent_folder / f"{sample_id}.{extension}"
            meta = torchaudio.info(path)

            speaker_id, chapter_id, utterance_id = sample_id.split("-")
            sample_id = f"ls/{speaker_id}/{chapter_id}/{utterance_id}"

            speaker_df = df_speaker_meta.loc[df_speaker_meta["id"] == int(speaker_id)]
            gender = speaker_df["sex"].item().lower()

            sample = ShardCsvSample(
                key=sample_id,
                path=str(path),
                num_frames=meta.num_frames,
                sample_rate=meta.sample_rate,
                transcription=transcription,
                speaker_id=f"ls/{speaker_id}",
                recording_id=f"ls/{chapter_id}",
                gender=gender,
                language_tag="en",
            )

            all_samples.append(sample)

    return all_samples


########################################################################################
# entrypoint of script


@click.command()
@click.option(
    "--dir",
    "dir_path",
    type=pathlib.Path,
    required=True,
    help="path the root directory of librispeech split",
)
@click.option(
    "--csv",
    "csv_file",
    type=pathlib.Path,
    required=True,
    help="path to write output csv file to",
)
@click.option(
    "--speakers",
    "speaker_txt_file",
    type=pathlib.Path,
    required=True,
    help="path to text file containing metadata on speaker IDs",
)
@click.option(
    "--ext",
    "extension",
    type=str,
    default="wav",
    help="the extension used for each audio file",
)
def main(
    dir_path: pathlib.Path,
    csv_file: pathlib.Path,
    speaker_txt_file: pathlib.Path,
    extension: str,
):
    print(f"Generating {str(csv_file)} with data in {dir_path}", flush=True)
    meta = load_speaker_meta(speaker_txt_file)

    samples_found = traverse_split(
        path=dir_path, df_speaker_meta=meta, extension=extension
    )

    with yaspin.yaspin(text=f"writing {csv_file}"):
        ShardCsvSample.to_csv(csv_file, samples_found)


if __name__ == "__main__":
    main()
