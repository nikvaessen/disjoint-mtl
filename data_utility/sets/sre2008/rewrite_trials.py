########################################################################################
#
# Rewrite trial file format
#
# Author(s): Nik Vaessen
########################################################################################
import pathlib

import click as click

########################################################################################
# entrypoint of script


@click.command()
@click.option("--in", "in_path", type=pathlib.Path, required=True)
@click.option("--out", "out_path", type=pathlib.Path, required=True)
def main(in_path: pathlib.Path, out_path: pathlib.Path):
    out_path.parent.mkdir(exist_ok=True)

    with in_path.open("r") as inf, out_path.open("w") as outf:
        for line in inf.readlines():
            line = line.strip()

            if len(line) > 0:
                a, b, gt = line.split(" ")
                outf.write(f"sre08/unk/unk/{a} sre08/unk/unk/{b} {gt}\n")


if __name__ == "__main__":
    main()
