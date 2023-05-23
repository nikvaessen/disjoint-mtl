########################################################################################
#
# Generate a sbatch file to parallelize transcribing the voxceleb dataset with SLURM
#
# Author(s): Nik Vaessen
########################################################################################

import pathlib
from typing import Optional

import click

########################################################################################
# script


@click.command()
@click.option("--dir", type=pathlib.Path, required=True)
@click.option("--out", type=pathlib.Path, required=True)
@click.option("--cache", type=pathlib.Path, default=None)
@click.option("--model", type=str, required=True)
def main(
    dir: pathlib.Path,
    out: pathlib.Path,
    model: str,
    cache: Optional[pathlib.Path] = None,
):
    shards = sorted([x for x in dir.rglob("*.tar*") if x.is_file()])

    file = f"""#!/usr/bin/env bash
#SBATCH --partition=das
#SBATCH --account=das
#SBATCH --array=0-{len(shards)-1}%1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --time=14-10:00:00
#SBATCH --output=./logs/{out.stem}_%A_%a.out
#SBATCH --error=./logs/{out.stem}_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
"""

    for idx, shard in enumerate(shards):
        file += f"""
if [ "$SLURM_ARRAY_TASK_ID" = {idx} ]
then
    poetry run transcribe_whisper {str(shard.absolute())} --model {model} {f'--cache {str(cache.absolute())}' if cache is not None else ''}
fi
"""

    with out.open("w") as f:
        f.write(file)


if __name__ == "__main__":
    main()
