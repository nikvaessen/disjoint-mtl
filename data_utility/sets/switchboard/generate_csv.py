########################################################################################
#
# This script can be used to generate the CSV file(s) which will be used to write
# the switchboard dataset into shards.
#
# Author(s): Nik Vaessen
########################################################################################

import pathlib


import click
import jiwer.transforms
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
        skiprows=0,
        header=None,
        names=[
            "id",
            "pin",
            "target",
            "sex",
            "birth_year",
            "dialect_area",
            "education",
            "ti",
            "payment_type",
            "amt_pd",
            "con",
            "remarks",
            "calls_deleted",
            "speaker_partition",
        ],
        sep=r",\s+",
        engine="python",
    )

    return df


def maybe_fix_transcription(sentence: str):
    modified = False

    if "{speaking on another phone}" in sentence:
        sentence = sentence.replace("{speaking on another phone}", " ")
        modified = True

    if "3e" in sentence:
        sentence = sentence.replace("3e", "e")
        modified = True

    if "uh-huh" in sentence:
        sentence = sentence.replace("uh-huh", " ")
        modified = True

    if "uh" in sentence:
        sentence = sentence.replace("uh", " ")
        modified = True

    for punc in ['"', "-", ".", ",", "!", "?", "#", "(", ")"]:
        if punc in sentence:
            sentence = sentence.replace(punc, " ")
            modified = True

    if modified:
        sentence = jiwer.transforms.RemoveMultipleSpaces()(sentence)
        sentence = sentence.strip()

    return sentence


def traverse_split(path: pathlib.Path, df_speaker_meta: pd.DataFrame, extension: str):
    all_samples = []

    # we find each 'transcript.txt` file, and manually determine the path to
    # the audio file for each transcription
    with yaspin.yaspin(text=f"recursively globbing {path}"):
        files = [f for f in path.rglob("transcript.txt")]

    for trans_file in tqdm(files):
        parent_folder = trans_file.parent

        # load transcriptions
        with trans_file.open("r") as f:
            lines = f.readlines()

        # split transcriptions into ID and sentence
        samples_in_file = [line.strip().lower().split("\t") for line in lines]
        samples_in_file = [(line[0], line[1]) for line in samples_in_file]

        for sample_id, transcription in samples_in_file:
            path = parent_folder / f"{sample_id}.{extension}"
            meta = torchaudio.info(path)

            speaker_id = path.parent.parent.name
            chapter_id = path.parent.name
            utterance_id = path.stem
            sample_id = f"sw/{speaker_id}/{chapter_id}/{utterance_id}"

            speaker_df = df_speaker_meta.loc[df_speaker_meta["id"] == int(speaker_id)]
            gender = speaker_df["sex"].item().lower()

            if gender == '"male"':
                gender = "m"
            elif gender == '"female"':
                gender = "f"
            else:
                raise ValueError(f"unknown gender {gender}")

            transcription = maybe_fix_transcription(transcription)

            if len(transcription) == 0:
                continue

            sample = ShardCsvSample(
                key=sample_id,
                path=str(path),
                num_frames=meta.num_frames,
                sample_rate=meta.sample_rate,
                transcription=transcription,
                speaker_id=f"sw/{speaker_id}",
                recording_id=f"sw/{chapter_id}",
                gender=gender,
            )

            all_samples.append(sample)

    return ShardCsvSample.to_dataframe(all_samples)


########################################################################################
# entrypoint of script


@click.command()
@click.option(
    "--dir",
    "dir_path",
    type=pathlib.Path,
    required=True,
    help="path the root directory of chunked version of switchboard",
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
    help="path to database file containing metadata on speaker IDs",
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

    df_found = traverse_split(path=dir_path, df_speaker_meta=meta, extension=extension)

    with yaspin.yaspin(text=f"writing {csv_file}"):
        csv_file.parent.mkdir(exist_ok=True, parents=True)
        df_found.to_csv(str(csv_file), index=False)


if __name__ == "__main__":
    main()
