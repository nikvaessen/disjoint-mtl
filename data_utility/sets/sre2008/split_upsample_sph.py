########################################################################################
#
# Split a folder of .sph files with 2-channel audio into separate files and optionally
# upsample from 8 to 16khz audio
#
# Author(s): Nik Vaessen
########################################################################################

import pathlib

import click
import torchaudio
import tqdm

# torchaudio.set_audio_backend("soundfile")

########################################################################################
# entrypoint of script


@click.command()
@click.option("--in", "in_path", required=True, type=pathlib.Path)
@click.option("--out", "out_path", required=True, type=pathlib.Path)
def main(in_path: pathlib.Path, out_path: pathlib.Path):
    files = [f for f in in_path.rglob("*.sph")]
    out_path.mkdir(exist_ok=True)

    for f in tqdm.tqdm(files):
        tensor, sr = torchaudio.load(f)

        if sr != 16_000:
            tensor = torchaudio.functional.resample(
                tensor, orig_freq=sr, new_freq=16_000
            )

        spk_a = tensor[0:1, :]
        spk_b = tensor[1:2, :]

        out_a = out_path / f"{f.stem}-a.wav"
        out_b = out_path / f"{f.stem}-b.wav"

        torchaudio.save(out_a, spk_a, 16_000, encoding="PCM_S", format="wav")
        torchaudio.save(out_b, spk_b, 16_000, encoding="PCM_S", format="wav")


if __name__ == "__main__":
    main()
