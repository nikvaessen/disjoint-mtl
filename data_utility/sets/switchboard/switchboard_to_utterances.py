########################################################################################
#
# Enable switchboard to be transformed into a voxceleb & librispeech like dataset,
# where we have a folder structure <speaker_id>/<session_id>/<utterance_d>.
#
# To enable this, this script chunks each phone call into single utterances for each
# individual speaker, before recreating a speaker-based directory structure.
#
# Author(s): Nik Vaessen
########################################################################################

import math
import pathlib

from collections import defaultdict
from functools import lru_cache
from itertools import chain
from shutil import rmtree
from typing import List, Dict

import click
import torchaudio


########################################################################################
# logic for loading files


@lru_cache(1)
def load_reverse_speaker_ids(chan_err_path: pathlib.Path):
    with chan_err_path.open("r") as f:
        id_lines = [
            int(line.strip()) for line in f.readlines() if not line.startswith("#")
        ]

    return id_lines


@lru_cache(1)
def load_no_trans_file(no_trans_path: pathlib.Path):
    with no_trans_path.open("r") as f:
        id_lines = [
            int(line.strip()) for line in f.readlines() if not line.startswith("#")
        ]

    return id_lines


def has_trans_file(file_id: int, no_trans_path: pathlib.Path):
    no_trans_id = load_no_trans_file(no_trans_path)

    return file_id not in no_trans_id


def is_reverse_spk_id(file_id: int, chan_err_path: pathlib.Path):
    reverse_ids = load_reverse_speaker_ids(chan_err_path)

    return file_id in reverse_ids


def extract_spk_id(txt_file: pathlib.Path, chan_err_file: pathlib.Path):
    with txt_file.open("r") as f:
        lines = f.readlines()
        filename_line = lines[0].lower()

        if "filename:" in filename_line:
            file_id, spk_a_id, spk_b_id = filename_line.split(":")[1].strip().split("_")
        else:
            raise ValueError(f"{txt_file} is broken")

    spk_a_id = spk_a_id.replace("b", "")
    spk_a_id = spk_a_id.replace("l", "1")
    spk_b_id = spk_b_id.replace("l", "1")

    assert len(file_id) == len(spk_a_id) == len(spk_b_id) == 4

    file_id = int(file_id)

    spk_a_id = int(spk_a_id)
    spk_b_id = int(spk_b_id)

    if is_reverse_spk_id(file_id, chan_err_file):
        return file_id, spk_b_id, spk_a_id
    else:
        return file_id, spk_a_id, spk_b_id


def find_txt_label_file(root_dir: pathlib.Path):
    label_dir = root_dir / "trans"

    if not label_dir.exists():
        raise ValueError(f"expected {label_dir} to exist")

    files = [f for f in label_dir.rglob("sw*.txt")]

    return files


def find_mrk_label_file(root_dir: pathlib.Path):
    label_dir = root_dir / "trans"

    if not label_dir.exists():
        raise ValueError(f"expected {label_dir} to exist")

    files = [f for f in label_dir.rglob("sw*.mrk")]

    return files


def find_wav_file(root_dir: pathlib.Path):
    wav_dir = root_dir / "data"

    if not wav_dir.exists():
        raise ValueError(f"expected {wav_dir} to exist")

    files = [f for f in wav_dir.rglob("sw*.wav")]

    return files


def extract_file_id(path: pathlib.Path):
    path_str = str(path.name)

    if path_str.startswith("sw"):
        file_id = int(path_str[2:6])

        return file_id
    else:
        raise ValueError(f"unable to parse {path}")


def match_files(
    wav_files: List[pathlib.Path],
    txt_files: List[pathlib.Path],
    mrk_files: List[pathlib.Path],
):
    cache = defaultdict(dict)

    for f in chain(wav_files, txt_files, mrk_files):
        file_id = extract_file_id(f)
        ext = f.suffix

        if ext == ".wav":
            cache[file_id]["wav"] = f
        elif ext == ".txt":
            cache[file_id]["txt"] = f
        elif ext == ".mrk":
            cache[file_id]["mrk"] = f
        else:
            raise ValueError(f"unknown extension {ext=}")

    cache = [v for k, v in sorted(cache.items(), key=lambda t: t[0]) if len(v) == 3]

    return cache


########################################################################################
# logic for chunking audio into sentences


def mrk_file_to_sentence_list(mrk_file: pathlib.Path, chan_err_path: pathlib.Path):
    with mrk_file.open("r") as f:
        lines = [line.strip() for line in f.readlines()]
        lines = [line for line in lines if len(line) > 1]

    file_id = extract_file_id(mrk_file)
    is_reversed = is_reverse_spk_id(file_id, chan_err_path)

    # convert raw strings to tuple of (sentence ID, start time, duration, keyword)
    def line_to_tuple(line):
        line_split = " ".join(line.split())

        tpl = [split for split in line_split.split(" ") if split != ""]

        if len(tpl) == 5:
            print(f"`{tpl}` `{line}` `{line_split}`")
            tpl[3] = " ".join(tpl[3:])
            del tpl[4]

        if len(tpl) != 4:
            print(f"`{tpl}` `{line}` `{line_split}`")

        assert len(tpl) == 4

        if "@" in tpl[0]:
            # remove @ or @@
            tpl[0] = tpl[0].removeprefix("@")
            tpl[0] = tpl[0].removeprefix("@")

        if "*" in tpl[0]:
            # remove @ or @@
            tpl[0] = tpl[0].removeprefix("*")
            tpl[0] = tpl[0].removeprefix("*")

        if is_reversed:
            sid = tpl[0]

            if len(sid) == 0:
                new_sid = sid
            elif sid[0] == "A":
                new_sid = "B" + sid[1:]
            elif sid[0] == "B":
                new_sid = "A" + sid[1:]
            else:
                raise ValueError("confused")

            tpl[0] = new_sid

        return tuple(tpl)

    lines = [line_to_tuple(line) for line in lines]

    # split into separate sentences
    sentence_dict = defaultdict(list)

    for line in lines:
        sentence_id, start_time, duration, keyword = line

        try:
            start_time = float(start_time)
            duration = float(duration)
        except ValueError:
            continue

        if "[" in keyword or "]" in keyword:
            continue

        if len(sentence_id) == 0:
            continue

        sentence_id_split = sentence_id.split(".")
        sentence_id_speaker = sentence_id_split[0]
        sentence_id_int = sentence_id_split[1]

        sentence_dict[sentence_id_int].append(
            (sentence_id_speaker, sentence_id_int, start_time, duration, keyword)
        )

    sentence_list = []
    for k in sorted(sentence_dict.keys(), key=lambda x: int(x)):
        sentence = sorted(sentence_dict[k], key=lambda tpl: int(tpl[1]))

        sentence_list.append(sentence)

    return sentence_list


def split_into_chunks(
    data_triplets: List[Dict[str, pathlib.Path]],
    channel_correction_file: pathlib.Path,
    out_dir: pathlib.Path,
    min_duration_sec: float,
    max_duration_sec: float,
):
    for triplet in data_triplets:
        wav = triplet["wav"]
        mrk = triplet["mrk"]
        txt = triplet["txt"]

        print(f"analysing {wav} with {mrk}")
        wav_tensor, sr = torchaudio.load(wav)

        try:
            file_id, speaker_a, speaker_b = extract_spk_id(txt, channel_correction_file)
        except AssertionError:
            print(f"skipped {txt} due to error")
            continue

        transcript_speaker_a = (
            out_dir / f"{speaker_a}" / f"{file_id}" / f"transcript.txt"
        )
        transcript_speaker_b = (
            out_dir / f"{speaker_b}" / f"{file_id}" / f"transcript.txt"
        )
        transcript_speaker_a.unlink(missing_ok=True)
        transcript_speaker_b.unlink(missing_ok=True)

        print(f"{file_id=} {speaker_a=} {speaker_b=}")

        sentences = mrk_file_to_sentence_list(mrk, channel_correction_file)

        for sentence_tuple_list in sentences:
            assert len(sentence_tuple_list) >= 1
            assert len(set(t[0] for t in sentence_tuple_list)) == 1
            assert len(set(t[1] for t in sentence_tuple_list)) == 1

            speaker_char = sentence_tuple_list[0][0]
            sentence_id = int(sentence_tuple_list[0][1])

            start_time = sentence_tuple_list[0][2]
            end_time = sentence_tuple_list[-1][2] + sentence_tuple_list[-1][3]
            duration = end_time - start_time

            if start_time < 0:
                continue

            if duration < min_duration_sec or duration > max_duration_sec:
                continue

            try:
                if speaker_char.lower() == "a":
                    sentence_tensor = seek_sentence_in_wav(
                        wav_tensor[0:1, :], sr, start_time, end_time
                    )
                    speaker_id = speaker_a
                    trans_file = transcript_speaker_a
                elif speaker_char.lower() == "b":
                    sentence_tensor = seek_sentence_in_wav(
                        wav_tensor[1:2, :], sr, start_time, end_time
                    )
                    speaker_id = speaker_b
                    trans_file = transcript_speaker_b
                else:
                    raise ValueError(f"unparseable {speaker_char=}")
            except ValueError:
                continue

            utt_id = f"{sentence_id:>06}"
            audio_path = out_dir / f"{speaker_id}" / f"{file_id}" / f"{utt_id}.wav"

            audio_path.parent.mkdir(exist_ok=True, parents=True)

            if sentence_tensor.shape[1] == 0:
                continue

            sentence_tensor = torchaudio.functional.resample(
                sentence_tensor, orig_freq=sr, new_freq=16_000
            )
            torchaudio.save(audio_path, sentence_tensor, 16_000)

            with trans_file.open("a") as f:
                sentence_str = " ".join([t[-1] for t in sentence_tuple_list])
                sentence_str = sentence_str.strip()

                f.write(f"{utt_id}\t{sentence_str}\n")


def seek_sentence_in_wav(
    wav_tensor, sample_rate: int, start_time_sec: float, end_time: float
):
    start_idx = math.floor(sample_rate * start_time_sec)
    end_idx = math.ceil(sample_rate * end_time)

    assert start_idx < end_idx
    sentence_tensor = wav_tensor[:, start_idx:end_idx]

    if sentence_tensor.shape[1] != end_idx - start_idx:
        raise ValueError("unable to extract given window")

    return sentence_tensor


########################################################################################
# script entrypoint


@click.command()
@click.option(
    "--dir",
    "root_dir",
    type=pathlib.Path,
    required=True,
    help="root directory of switchboard dataset",
)
@click.option(
    "--out",
    "output_dir_chunks",
    type=pathlib.Path,
    required=True,
    help="directory to write chunked version of dataset in",
)
@click.option(
    "--min",
    "min_duration_sec",
    type=float,
    default=3,
    help="min duration of a valid chunk in seconds",
)
@click.option(
    "--max",
    "max_duration_sec",
    type=float,
    default=20,
    help="max duration of a valid chunk in seconds",
)
@click.option(
    "--ses",
    "min_sessions",
    type=int,
    default=2,
    help="minimum number of sessions a speaker needs to have in order to be included",
)
def main(
    root_dir: pathlib.Path,
    output_dir_chunks: pathlib.Path,
    min_duration_sec: float,
    max_duration_sec: float,
    min_sessions: int,
):
    # load all input files
    channel_correction_path = root_dir / "chan_err.cnv"

    wav_files = find_wav_file(root_dir)
    mrk_files = find_mrk_label_file(root_dir)
    txt_files = find_txt_label_file(root_dir)

    data_triplets = match_files(wav_files, txt_files, mrk_files)

    split_into_chunks(
        data_triplets,
        channel_correction_path,
        output_dir_chunks,
        min_duration_sec,
        max_duration_sec,
    )

    # clean up the chunked dataset by
    # 1) verify each session has at least 1 utterance
    # 2) the number of transcripts matches the number of utterances
    # 3) remove each speaker folder with less than the minimum number of sessions
    for child_dir in [c for c in output_dir_chunks.iterdir()]:
        if not child_dir.is_dir():
            raise ValueError(f"{child_dir} is not a directory")

        num_sessions = 0
        for session in child_dir.iterdir():
            if not session.is_dir():
                raise ValueError(f"{session} is not a directory")

            transcript_file = session / "transcript.txt"

            if not transcript_file.exists():
                raise ValueError(f"unable to find {transcript_file}")

            with transcript_file.open("r") as f:
                tab = "\t"
                audio_files = [
                    session / f"{line.split(tab)[0]}.wav" for line in f.readlines()
                ]

                for af in audio_files:
                    if not af.exists():
                        raise ValueError(f"unable to find {af}")

            num_files = len(audio_files) + 1
            assert len(audio_files) >= 1
            assert num_files == len([f for f in session.iterdir()])

            num_sessions += 1

        if num_sessions < min_sessions:
            print(child_dir, f"has only {num_sessions} session, deleting...")
            rmtree(str(child_dir))


if __name__ == "__main__":
    main()
