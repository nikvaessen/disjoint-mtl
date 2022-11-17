########################################################################################
#
# This file implement a datamodule for a MTL dataset with disjoint data
# by wrapping the speaker-only and speech-only data modules.
#
# Author(s): Nik Vaessen
########################################################################################

import json
import pathlib

from dataclasses import dataclass
from typing import Optional, List

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from data_utility.eval.speaker.evaluator import SpeakerTrial
from data_utility.pipe.builder import (
    SpeakerRecognitionDataPipeBuilder,
    SpeechRecognitionDataPipeBuilder,
)

from src.data.module import (
    SpeechRecognitionDataModuleConfig,
    SpeakerRecognitionDataModuleConfig,
)
from src.util.config_util import CastingConfig


########################################################################################
# config


@dataclass
class DisjointMTLDataModuleConfig(CastingConfig):
    speech_dm_config: SpeechRecognitionDataModuleConfig
    speaker_dm_config: SpeakerRecognitionDataModuleConfig


########################################################################################
# implementation


class DisjointMTLDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: DisjointMTLDataModuleConfig,
        speaker_train_pipe_builder: SpeakerRecognitionDataPipeBuilder,
        speaker_val_pipe_builder: SpeakerRecognitionDataPipeBuilder,
        speaker_test_pipe_builder: SpeakerRecognitionDataPipeBuilder,
        speech_train_pipe_builder: SpeechRecognitionDataPipeBuilder,
        speech_val_pipe_builder: SpeechRecognitionDataPipeBuilder,
        speech_test_pipe_builder: SpeechRecognitionDataPipeBuilder,
    ):
        super(DisjointMTLDataModule, self).__init__()

        self.cfg = cfg

        self.speaker_train_pipe_builder = speaker_train_pipe_builder
        self.speaker_val_pipe_builder = speaker_val_pipe_builder
        self.speaker_test_pipe_builder = speaker_test_pipe_builder

        self.speech_train_pipe_builder = speech_train_pipe_builder
        self.speech_val_pipe_builder = speech_val_pipe_builder
        self.speech_test_pipe_builder = speech_test_pipe_builder

        # set _num_speakers and set speaker_to_idx on pipe builders
        self._num_speakers = None
        self._init_speakers()

        if not (
            len(self.cfg.test_names)
            == len(self.cfg.test_shards)
            == len(self.cfg.test_trials)
        ):
            raise ValueError("length of test names, shards, and trials does not match")

        # init in setup()
        self.train_dp = None
        self.val_dp = None
        self.test_dp_list = None

    def _load_speakers_json(self):
        with self.cfg.speaker_json.open("r") as f:
            return json.load(f)

    def _init_speakers(self):
        assert isinstance(self.train_pipe_builder, SpeakerRecognitionDataPipeBuilder)
        assert isinstance(self.val_pipe_builder, SpeakerRecognitionDataPipeBuilder)
        assert isinstance(self.test_pipe_builder, SpeakerRecognitionDataPipeBuilder)

        speaker_to_idx = self._load_speakers_json()["speaker_to_idx"]

        self._num_speakers = len(speaker_to_idx)
        self.train_pipe_builder.set_speaker_to_idx(speaker_to_idx)
        self.val_pipe_builder.set_speaker_to_idx(speaker_to_idx)

    def get_num_train_speakers(self) -> int:
        return self._num_speakers

    def get_test_speaker_eval_list(self) -> List[List[SpeakerTrial]]:
        return [SpeakerTrial.from_file(f) for f in self.cfg.test_trials]

    def get_test_names(self):
        return self.cfg.test_names

    def setup(self, stage: Optional[str] = None) -> None:
        # train dp
        self.train_dp = self.train_pipe_builder.get_pipe(
            shard_dirs=self.cfg.train_shard_paths,
            shard_file_pattern=self.cfg.shard_file_pattern,
        )

        # val dp
        self.val_dp = self.val_pipe_builder.get_pipe(
            shard_dirs=self.cfg.val_shard_paths,
            shard_file_pattern=self.cfg.shard_file_pattern,
        )

        # test dp
        self.test_dp_list = [
            self.test_pipe_builder.get_pipe(
                shard_dirs=path, shard_file_pattern=self.cfg.shard_file_pattern
            )
            for path in self.cfg.test_shards
        ]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_pipe_builder.wrap_pipe(self.train_dp)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_pipe_builder.wrap_pipe(self.val_dp)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return [self.test_pipe_builder.wrap_pipe(dp) for dp in self.test_dp_list]
