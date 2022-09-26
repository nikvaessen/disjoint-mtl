########################################################################################
#
# This file implement a datamodule for a disjointed LibriSpeech dataset,
# where 1 part only has speaker labels, and 1 part only has speech labels.
#
# Author(s): Nik Vaessen
########################################################################################

import json
import pathlib

from dataclasses import dataclass
from typing import Optional, Tuple, List

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from data_utility.eval.speaker.evaluator import SpeakerTrial
from data_utility.pipe.builder import (
    DataPipeBuilder,
    SpeakerRecognitionDataPipeBuilder,
    SpeechRecognitionDataPipeBuilder,
)
from src.util.config_util import CastingConfig


########################################################################################
# config


@dataclass
class DisjointedLibriSpeechDataModuleConfig(CastingConfig):
    # path to folders containing train, val and test shards
    train_set1_shard_path: pathlib.Path
    train_set2_shard_path: pathlib.Path

    val_set1_shard_path: pathlib.Path
    val_set2_shard_path: pathlib.Path

    dev_clean_shard_path: pathlib.Path
    dev_other_shard_path: pathlib.Path

    test_clean_shard_path: pathlib.Path
    test_other_shard_path: pathlib.Path

    # shard pattern
    shard_file_pattern: str

    # path to meta files for speaker info
    disjoint1_speaker_json: pathlib.Path
    disjoint2_speaker_json: pathlib.Path
    dev_speaker_json: pathlib.Path
    test_speaker_json: pathlib.Path

    # path to meta file for speech info
    char_vocab_json: pathlib.Path

    # path to trials for speaker recognition eval
    dev_clean_trial_path: pathlib.Path
    dev_other_trial_path: pathlib.Path
    test_clean_trial_path: pathlib.Path
    test_other_trial_path: pathlib.Path

    # train set options
    # set1: for the ASR task
    # set2: for the speech task
    # mtl: for doing MTL with disjointed dataset
    train_set_mode: str

    # which pipes to use
    task_mode: str  # 'speech', 'speaker' or 'mtl'


########################################################################################
# implementation


class LibriSpeechDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: DisjointedLibriSpeechDataModuleConfig,
        speech_pipe_builders: Tuple[DataPipeBuilder, DataPipeBuilder, DataPipeBuilder],
        speaker_pipe_builders: Tuple[DataPipeBuilder, DataPipeBuilder, DataPipeBuilder],
    ):
        super(LibriSpeechDataModule, self).__init__()

        self.cfg = cfg

        if self.cfg.task_mode == "speech":
            self.train_pipe_builder = speech_pipe_builders[0]
            self.val_pipe_builder = speech_pipe_builders[1]
            self.test_pipe_builder = speech_pipe_builders[2]
            self._set_character_to_idx()

        elif self.cfg.task_mode == "speaker":
            self.train_pipe_builder = speaker_pipe_builders[0]
            self.val_pipe_builder = speaker_pipe_builders[1]
            self.test_pipe_builder = speaker_pipe_builders[2]
            self._set_speaker_to_idx()

        elif self.cfg.task_mode == "mtl":
            raise NotImplemented()

        else:
            raise ValueError(f"unknown {self.cfg.task_mode=}")

        # init in setup()
        self.train_dp = None
        self.val_dp = None
        self.dev_dp_clean = None
        self.dev_dp_other = None
        self.test_dp_clean = None
        self.test_dp_other = None

    def _train_speakers_json(self):
        if self.cfg.train_set_mode == "set1":
            with self.cfg.disjoint1_speaker_json.open("r") as f:
                return json.load(f)
        elif self.cfg.train_set_mode == "set2":
            with self.cfg.disjoint2_speaker_json.open("r") as f:
                return json.load(f)
        else:
            raise ValueError("cannot load correct train_speaker_json")

    def _dev_speakers_json(self):
        with self.cfg.dev_speaker_json.open("r") as f:
            return json.load(f)

    def _test_speakers_json(self):
        with self.cfg.test_speaker_json.open("r") as f:
            return json.load(f)

    def _character_vocab_json(self):
        with self.cfg.char_vocab_json.open("r") as f:
            return json.load(f)

    def _set_speaker_to_idx(self):
        assert isinstance(self.train_pipe_builder, SpeakerRecognitionDataPipeBuilder)
        assert isinstance(self.val_pipe_builder, SpeakerRecognitionDataPipeBuilder)
        assert isinstance(self.test_pipe_builder, SpeakerRecognitionDataPipeBuilder)

        self.train_pipe_builder.set_speaker_to_idx(
            self._train_speakers_json()["speaker_to_idx"]
        )
        self.val_pipe_builder.set_speaker_to_idx(
            self._train_speakers_json()["speaker_to_idx"]
        )
        self.test_pipe_builder.set_speaker_to_idx(
            self._dev_speakers_json()["speaker_to_idx"]
            | self._test_speakers_json()["speaker_to_idx"]
        )

    def _set_character_to_idx(self):
        assert isinstance(self.train_pipe_builder, SpeechRecognitionDataPipeBuilder)
        assert isinstance(self.val_pipe_builder, SpeechRecognitionDataPipeBuilder)
        assert isinstance(self.test_pipe_builder, SpeechRecognitionDataPipeBuilder)

        self.train_pipe_builder.set_char_to_idx(
            self._character_vocab_json()["char_to_idx"]
        )
        self.val_pipe_builder.set_char_to_idx(
            self._character_vocab_json()["char_to_idx"]
        )
        self.test_pipe_builder.set_char_to_idx(
            self._character_vocab_json()["char_to_idx"]
        )

    def get_num_train_speakers(self) -> int:
        return len(self._train_speakers_json()["speakers"])

    def get_test_speaker_eval_list(self) -> List[List[SpeakerTrial]]:
        return [
            SpeakerTrial.from_file(f)
            for f in [
                self.cfg.dev_clean_trial_path,
                self.cfg.dev_other_trial_path,
                self.cfg.test_clean_trial_path,
                self.cfg.test_other_trial_path,
            ]
        ]

    def get_test_names(self):
        return ["dev_clean", "dev_other", "test_clean", "test_other"]

    def get_vocab_size(self):
        return len(self._character_vocab_json()["characters"])

    def get_idx_to_char(self):
        vocab_json = self._character_vocab_json()

        idx_to_char = {int(k): v for k, v in vocab_json["idx_to_char"].items()}

        return idx_to_char

    def setup(self, stage: Optional[str] = None) -> None:
        # train dp
        if self.cfg.train_set_mode == "set1" and self.cfg.task_mode == "speech":
            self.train_dp = self.train_pipe_builder.get_pipe(
                shard_dirs=self.cfg.train_set1_shard_path,
                shard_file_pattern=self.cfg.shard_file_pattern,
            )
            self.val_dp = self.val_pipe_builder.get_pipe(
                shard_dirs=self.cfg.val_set1_shard_path,
                shard_file_pattern=self.cfg.shard_file_pattern,
            )
        elif self.cfg.train_set_mode == "set2" and self.cfg.task_mode == "speaker":
            self.train_dp = self.train_pipe_builder.get_pipe(
                shard_dirs=self.cfg.train_set2_shard_path,
                shard_file_pattern=self.cfg.shard_file_pattern,
            )
            self.val_dp = self.val_pipe_builder.get_pipe(
                shard_dirs=self.cfg.val_set2_shard_path,
                shard_file_pattern=self.cfg.shard_file_pattern,
            )
        elif self.cfg.train_set_mode == "mtl":
            raise NotImplemented()
        else:
            raise ValueError(
                f"unknown combination of {self.cfg.train_set_mode=} and {self.cfg.task_mode=}"
            )

        # test dp
        self.dev_dp_clean = self.test_pipe_builder.get_pipe(
            shard_dirs=self.cfg.dev_clean_shard_path,
            shard_file_pattern=self.cfg.shard_file_pattern,
        )
        self.dev_dp_other = self.test_pipe_builder.get_pipe(
            shard_dirs=self.cfg.dev_other_shard_path,
            shard_file_pattern=self.cfg.shard_file_pattern,
        )
        self.test_dp_clean = self.test_pipe_builder.get_pipe(
            shard_dirs=self.cfg.test_clean_shard_path,
            shard_file_pattern=self.cfg.shard_file_pattern,
        )
        self.test_dp_other = self.test_pipe_builder.get_pipe(
            shard_dirs=self.cfg.test_other_shard_path,
            shard_file_pattern=self.cfg.shard_file_pattern,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_pipe_builder.wrap_pipe(self.train_dp)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_pipe_builder.wrap_pipe(self.val_dp)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return [
            self.test_pipe_builder.wrap_pipe(self.dev_dp_clean),
            self.test_pipe_builder.wrap_pipe(self.dev_dp_other),
            self.test_pipe_builder.wrap_pipe(self.test_dp_clean),
            self.test_pipe_builder.wrap_pipe(self.test_dp_other),
        ]
