########################################################################################
#
# This script converts the trial list of voxceleb2 to the general format.
#
# Author(s): Nik Vaessen
########################################################################################

import pathlib

import click

from data_utility.eval.speaker.evaluator import SpeakerTrial


########################################################################################
# logic for converting


def convert_trial_list(in_path: pathlib.Path, out_path: pathlib.Path, ds: str):
    with in_path.open("r") as f:
        lines = [line.strip() for line in f.readlines()]

    trials = []

    for line in lines:
        equal, left, right = line.split(" ")

        left = f"{ds}/{left}"
        left = left.removesuffix(".wav")
        right = f"{ds}/{right}"
        right = right.removesuffix(".wav")

        trials.append(
            SpeakerTrial(left=left, right=right, same_speaker=bool(int(equal)))
        )

    SpeakerTrial.to_file(out_path, trials)


########################################################################################
# entrypoint of script


@click.command()
@click.option("--path", type=pathlib.Path, required=True)
@click.option("--out", type=pathlib.Path, required=True)
@click.option("--ds", type=click.Choice(["vc1", "vc2"]), required=True)
def main(path: pathlib.Path, out: pathlib.Path, ds: str):
    print(f"converting {str(path)} to {str(out)} by adding dataset id {ds}")
    convert_trial_list(path, out, ds)


if __name__ == "__main__":
    main()
