########################################################################################
#
# This file implement a datamodule for the LibriSpeech dataset.
#
# Author(s): Nik Vaessen
########################################################################################

import pathlib

from dataclasses import dataclass

from src.util.config_util import CastingConfig


########################################################################################
# config


@dataclass
class LibriSpeechDataModuleConfig(CastingConfig):
    # path to folders containing train, val and test shards
    train_c100_shard_path: pathlib.Path
    train_c360_shard_path: pathlib.Path
    train_o500_shard_path: pathlib.Path

    val_clean_shard_path: pathlib.Path
    val_other_shard_path: pathlib.Path

    test_clean_shard_path: pathlib.Path
    test_other_shard_path: pathlib.Path

    # path to character vocabulary
    vocabulary_json_path: pathlib.Path

    # path to disjoint split
    disjoint_split_json_path: pathlib.Path

    # path to trials for speaker recognition eval
    dev_clean_trial_path: pathlib.Path
    dev_other_trial_path: pathlib.Path
    test_clean_trial_path: pathlib.Path
    test_other_trial_path: pathlib.Path

    # train set options
    # 960h: clean-100, clean-360 and other-500 subsets
    # 100h: clean-100 subset
    # disjoint: 480 hours (set1)
    train_set_mode: str

    # pipe options
    task_mode: str  # 'speech' or 'speaker'


########################################################################################
# implementation


class LibriSpeechDataModule:
    def __init__(self, cfg=LibriSpeechDataModuleConfig):
        self.cfg = cfg
