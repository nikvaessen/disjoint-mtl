########################################################################################
#
# This file implement a datamodule for the LibriSpeech dataset.
#
# Author(s): Nik Vaessen
########################################################################################

import pathlib

from dataclasses import dataclass

from data_utility.pipe.builder import (
    DataPipeBuilder,
    SpeakerRecognitionDataPipeBuilder,
    SpeechRecognitionDataPipeBuilder,
)
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

    # path to meta files for speaker info
    train_speaker_json: pathlib.Path
    dev_speaker_json: pathlib.Path
    test_speaker_json: pathlib.Path

    # path to meta file for speech info
    char_vocab_json: pathlib.Path

    # disjoint split (480h version)
    disjoint_split_json: pathlib.Path

    # path to trials for speaker recognition eval
    dev_clean_trial_path: pathlib.Path
    dev_other_trial_path: pathlib.Path
    test_clean_trial_path: pathlib.Path
    test_other_trial_path: pathlib.Path

    # train set options
    # 960h: clean-100, clean-360 and other-500 subsets
    # 100h: clean-100 subset
    # disjoint: 480 hours (set1 for speech, set2 for speaker)
    train_set_mode: str

    # pipe options
    task_mode: str  # 'speech' or 'speaker'


########################################################################################
# implementation


class LibriSpeechDataModule:
    def __init__(
            self,
            cfg: LibriSpeechDataModuleConfig,
            train_pipe_builder: DataPipeBuilder,
            val_pipe_builder: DataPipeBuilder,
            test_pipe_builder: DataPipeBuilder,
    ):
        self.cfg = cfg

        if self.cfg.task_mode == "speaker":
            assert [
                isinstance(x, SpeakerRecognitionDataPipeBuilder)
                for x in [train_pipe_builder, val_pipe_builder, test_pipe_builder]
            ]
        elif self.cfg.task_mode == "speech":
            assert [
                isinstance(x, SpeechRecognitionDataPipeBuilder)
                for x in [train_pipe_builder, val_pipe_builder, test_pipe_builder]
            ]
            pass
        else:
            raise ValueError(f"unknown {self.cfg.task_mode=}")

        self.train_pipe_builder = train_pipe_builder
        self.val_pipe_builder = val_pipe_builder
        self.test_pipe_builder = test_pipe_builder

    def num_train_speakers(self):
        pass

    def character_vocab(self):
        pass
