########################################################################################
#
# Various utility methods
#
# Author(s): Nik Vaessen
########################################################################################

import re

from data_utility.util.validate import KNOWN_DATASET_ID, check_key


########################################################################################
# utility related to the ID (key) of data samples


def sort_speaker_id_key(speaker_id: str):
    if speaker_id.startswith("ls/"):
        return int(speaker_id.split("/")[1])
    elif speaker_id.startswith("vc1") or speaker_id.startswith("vc2"):
        return int(speaker_id.split("/id")[1])
    elif speaker_id.startswith("sw"):
        return int(speaker_id.split("/")[1])
    else:
        raise ValueError(f"unsupported sorting of {speaker_id=}")


def split_sample_key(key: str):
    check_key(key)

    data_id, speaker_id, recording_id, utt_id = key.split("/")

    return data_id, speaker_id, recording_id, utt_id


def search_sample_key(potential_key: str):
    for data_id in KNOWN_DATASET_ID:
        result = re.search(rf"{data_id}\/(\w)+\/(\w)+\/(\w)+", potential_key)

        if result is not None:
            key = result.group()
            if isinstance(key, str):
                return key
            else:
                raise ValueError(f"multiple results: {result}")

    return None
