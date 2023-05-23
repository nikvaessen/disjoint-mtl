#! /usr/bin/env python3
########################################################################################
#
# Converts a set of files to wav by using FFMPEG.
#
# Author(s): Nik Vaessen
########################################################################################

import multiprocessing
import pathlib
import subprocess

import click

from tqdm import tqdm
from yaspin import yaspin

########################################################################################
# methods for converting


def subprocess_convert_to_wav(infile: str, outfile: str):
    """
    Use a subprocess calling FFMPEG to convert a file to 16 KHz .wav file.

    Parameters
    ----------
    infile: path to file which needs to be converted
    outfile: path where converted file needs to be stored
    """
    subprocess.check_output(
        [
            "ffmpeg",
            "-y",
            "-i",
            infile,
            "-ac",
            "1",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            outfile,
        ],
        stderr=subprocess.PIPE,
    )


def convert_to_wav(
    directory_path: pathlib.Path,
    audio_in_extension: str,
    audio_out_extension: str,
    num_workers: int = 1,
    delete_after_conversion: bool = False,
    overwrite: bool = False,
):
    """
    Convert all files ending with "audio_in_extension" in a specified directory
    to 16KHZ pcm_s16le wave file by using FFMPEG.

    Parameters
    ----------
    directory_path: the directory to scan for .m4a files to convert
    num_workers: the number of threads to use (in order to speed up conversions)
    delete_after_conversion: whether to remove the converted .m4a files after conversion has completed.
    audio_in_extension: the extension of the files which need to be converted
    audio_out_extension: the extension of the files after conversion
    overwrite: if a file with audio_out_extension already exists, overwrite it
    """
    if audio_in_extension == audio_out_extension:
        raise ValueError(
            f"--ext={audio_in_extension} cannot be equal to --out={audio_out_extension}"
        )
    if len(audio_in_extension) <= 0:
        raise ValueError(f"{audio_in_extension=} must be non-empty")
    if len(audio_out_extension) <= 0:
        raise ValueError(f"{audio_out_extension=} must be non-empty")

    # find all files in the train and test subdirectories
    all_in_files = []

    with yaspin(
        text=f"recursively finding all files in `{directory_path}` with pattern '*{audio_in_extension}'"
    ):
        all_in_files.extend(
            [f for f in directory_path.rglob(f"*{audio_in_extension}") if f.is_file()]
        )

    # use multiple workers to call FFMPEG and convert the .m4a files to .wav
    print(
        f"converting all files with pattern `{audio_in_extension}` in `{directory_path}` to wav"
    )
    with tqdm(total=len(all_in_files)) as p, multiprocessing.Pool(
        processes=num_workers
    ) as workers:
        for infile in sorted(all_in_files):
            outfile = infile.parent / (infile.stem + audio_out_extension)

            if outfile.exists():
                if overwrite:
                    outfile.unlink()
                else:
                    p.update(1)
                    continue

            def cb_success(_):
                p.update(1)

            def cb_error(e):
                print(e)

                if isinstance(e, subprocess.CalledProcessError):
                    print("stdout:\n", e.stdout.decode("utf-8"))

            workers.apply_async(
                subprocess_convert_to_wav,
                args=(str(infile), str(outfile)),
                callback=cb_success,
                error_callback=cb_error,
            )

        workers.close()
        workers.join()

    # optionally delete all original files
    if delete_after_conversion:
        for f in all_in_files:
            f.unlink()


########################################################################################
# script execution


@click.command(
    "Convert all audio files in a given root "
    "directory to 16 KHz mono pcm_s16le wav files"
)
@click.option(
    "--dir",
    "dir_path",
    type=pathlib.Path,
    help="directory containing the audio files to be converted",
    required=True,
)
@click.option(
    "--ext",
    "audio_extension",
    type=str,
    help="the extension of the audio file(s) which need to be converted",
    required=True,
)
@click.option(
    "--out",
    "extension_after_convert",
    type=str,
    help="the extension of the audio file after conversion",
    default=".wav",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    help="Number of threads to use for converting the dataset files.",
)
@click.option(
    "--delete",
    "delete_original",
    is_flag=True,
    help="delete the original files after they have been converted",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="overwrite a previously existing file with the extension_after_convert extension",
)
def main(
    dir_path: pathlib.Path,
    audio_extension: str,
    extension_after_convert: str,
    workers: int,
    delete_original: bool,
    overwrite: bool,
):
    convert_to_wav(
        directory_path=dir_path,
        audio_in_extension=audio_extension,
        audio_out_extension=extension_after_convert,
        num_workers=workers,
        delete_after_conversion=delete_original,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    main()
