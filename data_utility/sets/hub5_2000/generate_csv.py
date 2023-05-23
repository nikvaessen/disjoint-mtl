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
    help="path the root directory of segmented wav+txt folder of hub5",
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
    csv_file: pathlib.Path
):
    samples: List[ShardCsvSample] = []
    files = [f.absolute() for f in dir_path.rglob("*.wav") if f.is_file()]

    for audio_file in tqdm(files):
        meta = torchaudio.info(str(audio_file))

        recording_id = audio_file.parent.parent.name
        spk_id = f"{recording_id}_{audio_file.parent.name}"

        transcript_file = audio_file.parent / f"{audio_file.stem}.txt"
        with transcript_file.open('r') as f:
            lines = f.readlines()
            assert len(lines) == 1
            transcript = lines[0].strip()

        samples.append(
            ShardCsvSample(
                key=f"hub5/{spk_id}/{recording_id}/{audio_file.stem}",
                path=str(audio_file),
                num_frames=meta.num_frames,
                sample_rate=meta.sample_rate,
                speaker_id=f"hub5/{spk_id}",
                recording_id=f"hub5/{recording_id}",
                language_tag='en',
                transcription=transcript
            )
        )

    with yaspin.yaspin(text=f"writing {csv_file}"):
        csv_file.parent.mkdir(exist_ok=True, parents=True)
        ShardCsvSample.to_csv(csv_file, samples)


if __name__ == "__main__":
    main()
