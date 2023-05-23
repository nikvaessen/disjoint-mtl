########################################################################################
#
# Create segments based on the ground-truth transcription timestamps.
#
# Author(s): Nik Vaessen
########################################################################################

import math
import pathlib
import re

from typing import Tuple, List

import click

from tqdm import tqdm

import torchaudio


########################################################################################
# utility for handling reference file


def load_reference(reference_path: pathlib.Path):
    # load file
    with reference_path.open("r") as txt:
        lines = [ln.strip() for ln in txt.readlines()]
        lines = [ln for ln in lines if len(ln) > 0]

    lines = [ln for ln in lines if re.match(r"\d+\.\d+\s+\d+\.\d+\s+[AB]:.+", ln)]

    tuples = []
    for ln in lines:
        label, transcript = ln.split(":")

        start, end, channel = label.split(" ")
        transcript = normalize_transcript(transcript)

        if len(transcript) == 0 or transcript.count(" ") == 0:
            continue
        if "-" in transcript:  # transcript couldn't be normalized (around 10)
            continue

        tpl = (float(start), float(end), channel, transcript)
        tuples.append(tpl)

    return tuples


def normalize_transcript(transcript: str) -> str:
    # everything to lower case
    transcript = transcript.lower()

    # remove annotations
    transcript = remove_annotations(transcript)

    # and make sure there's no leading or trailing whitespace
    return transcript.strip()


def remove_annotations(transcript):
    # **&leaky**
    transcript = re.sub(r"\*\*&?.+?\*\*", "", transcript)

    # {breath}
    transcript = re.sub(r"\{.+?}", "", transcript)

    # [[laughing]]
    transcript = re.sub(r"\[\[.+?]]", "", transcript)

    # [distorted]
    transcript = re.sub(r"\[.+?]", "", transcript)

    # <spanish >
    transcript = re.sub(r"<.+?>", "", transcript)

    # & indicates names
    transcript = re.sub(r"&", "", transcript)

    # partially said words indicated with -
    transcript = re.sub(r"\w+?-", " ", transcript)

    # filler words indicated with %
    transcript = re.sub(r"%\w+", " ", transcript)

    # background noise ((planes))
    transcript = re.sub(r"\(\(.+?\)\)?", "", transcript)

    # remove punctuation
    transcript = transcript.replace("?", " ")
    transcript = transcript.replace(".", " ")
    transcript = transcript.replace(",", " ")
    transcript = transcript.replace("--", " ")
    transcript = transcript.replace("//", " ")

    # remove double spaces
    transcript = re.sub(r"\s{2,}", " ", transcript)
    transcript = transcript.strip()

    return transcript


########################################################################################
# Utility for segmenting audio file


def segment_utterances(
    out_dir: pathlib.Path,
    utt_file: pathlib.Path,
    segments: List[Tuple[float, float, str, str]],
):
    audio_tensor, sr = torchaudio.load(str(utt_file))
    segment_id = utt_file.stem.replace("_", "")

    if sr != 16_000:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 16_000)

    (out_dir / segment_id).mkdir(exist_ok=True, parents=True)
    with (out_dir / segment_id / "transcript.txt").open("w") as txt_file:
        for idx, segment in enumerate(segments):
            start_float, end_float, channel_str, transcript = segment

            start_idx = math.floor(start_float * 16_000)
            end_idx = math.ceil(end_float * 16_000)

            if channel_str == "A":
                channel = 0
            elif channel_str == "B":
                channel = 1
            else:
                raise ValueError(f"unknown {channel_str=}")

            audio_segment = audio_tensor[channel : channel + 1, start_idx:end_idx]
            segment_dir = out_dir / segment_id / channel_str.lower()
            segment_dir.mkdir(exist_ok=True, parents=True)

            torchaudio.save(
                segment_dir / f"{idx:06}.wav",
                audio_segment,
                16_000,
            )

            tr_file = segment_dir / f"{idx:06}.txt"
            with tr_file.open("w") as f:
                f.write(transcript)

            txt_file.write(f"{transcript}\n")


########################################################################################
# entrypoint of script


@click.command()
@click.option("--ref", "reference_dir", type=pathlib.Path, required=True)
@click.option("--utt", "utterance_dir", type=pathlib.Path, required=True)
@click.option("--out", type=pathlib.Path, required=True)
def main(reference_dir: pathlib.Path, utterance_dir: pathlib.Path, out: pathlib.Path):
    # load all reference and utterance files
    reference_files = [f for f in reference_dir.glob("*_*.txt") if f.is_file()]
    utterance_files = [f for f in utterance_dir.glob("*_*.sph") if f.is_file()]
    assert len(reference_files) == len(utterance_files) == 40

    # match utterance to reference
    reference_files = sorted(reference_files, key=lambda x: x.stem)
    utterance_files = sorted(utterance_files, key=lambda x: x.stem)

    # for each pairing, create the segmented utterance in the `out` directory
    for utt, ref in tqdm(
        zip(utterance_files, reference_files), total=len(reference_files)
    ):
        assert utt.stem == ref.stem

        label_tuples = load_reference(ref)
        segment_utterances(out, utt, label_tuples)


if __name__ == "__main__":
    main()
