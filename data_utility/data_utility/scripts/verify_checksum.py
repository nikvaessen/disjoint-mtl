#! /usr/bin/env python3
########################################################################################
#
# Verify that a given file adheres to a given checksum.
#
# Usage: ./verify_checksum <file_path> <checksum> --algo md5
#
# Author(s): Nik Vaessen
########################################################################################

import pathlib
import subprocess

import click

########################################################################################
# checksum algorithms


def md5hash(path: pathlib.Path):
    output_str = subprocess.check_output(["md5sum", str(path)])
    checksum = output_str.decode().split(" ")[0]

    return checksum


########################################################################################
# main logic


@click.command()
@click.argument(
    "file_path",
    type=pathlib.Path,
)
@click.argument(
    "checksum",
    type=str,
)
@click.option(
    "--algo",
    type=click.Choice(["md5"]),
    default="md5",
    help="Which checksum algorithm to use",
)
@click.option("--quit", is_flag=True, help="Disable string output")
def main(file_path: pathlib.Path, checksum: str, algo: str, quit: bool):
    if algo == "md5":
        actual_checksum = md5hash(file_path)
    else:
        raise ValueError(f"unsupported value for {algo=}")

    if actual_checksum == checksum:
        if not quit:
            print(f"{file_path} matched {checksum}")

        exit(0)
    else:
        print(
            f"{file_path} did not match {checksum}, actual checksum {actual_checksum}"
        )
        exit(1)


if __name__ == "__main__":
    main()
