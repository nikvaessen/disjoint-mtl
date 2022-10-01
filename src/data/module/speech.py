########################################################################################
#
# This file implement a datamodule for the LibriSpeech dataset.
#
# Author(s): Nik Vaessen
########################################################################################

import json
import pathlib

from dataclasses import dataclass
from typing import Optional, Tuple, List

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from data_utility.pipe.builder import SpeechRecognitionDataPipeBuilder
from src.util.config_util import CastingConfig


########################################################################################
# config


@dataclass
class SpeechRecognitionDataModuleConfig(CastingConfig):
    # name of dataset
    name: str

    # path to folder(s) containing train data
    train_shard_paths: List[pathlib.Path]

    # path to folder(s) containing val data
    val_shard_paths: List[pathlib.Path]

    # shard pattern
    shard_file_pattern: str

    # path to meta file for speech info
    char_vocab_json: pathlib.Path

    # name of each test set
    test_names: List[str]

    # path to each shard of test set (only 1 dir each)
    test_shards: List[pathlib.Path]


########################################################################################
# implementation


class SpeechRecognitionDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: SpeechRecognitionDataModuleConfig,
        train_pipe_builder: SpeechRecognitionDataPipeBuilder,
        val_pipe_builder: SpeechRecognitionDataPipeBuilder,
        test_pipe_builder: SpeechRecognitionDataPipeBuilder,
    ):
        super(SpeechRecognitionDataModule, self).__init__()

        self.cfg = cfg

        self.train_pipe_builder = train_pipe_builder
        self.val_pipe_builder = val_pipe_builder
        self.test_pipe_builder = test_pipe_builder

        # set _num_speakers and set speaker_to_idx on pipe builders
        self._vocab_size = None
        self._init_vocabulary()

        if not (len(self.cfg.test_names) == len(self.cfg.test_shards)):
            raise ValueError("length of test names and test shards does not match")

        # init in setup()
        self.train_dp = None
        self.val_dp = None
        self.test_dp_list = None

    def _load_character_vocab_json(self):
        with self.cfg.char_vocab_json.open("r") as f:
            return json.load(f)

    def _init_vocabulary(self):
        assert isinstance(self.train_pipe_builder, SpeechRecognitionDataPipeBuilder)
        assert isinstance(self.val_pipe_builder, SpeechRecognitionDataPipeBuilder)
        assert isinstance(self.test_pipe_builder, SpeechRecognitionDataPipeBuilder)

        char_to_idx = self._load_character_vocab_json()["char_to_idx"]

        self.train_pipe_builder.set_char_to_idx(char_to_idx)
        self.val_pipe_builder.set_char_to_idx(char_to_idx)
        self.test_pipe_builder.set_char_to_idx(char_to_idx)

        self._vocab_size = len(char_to_idx)

    def get_test_names(self):
        return self.cfg.test_names

    def get_vocab_size(self):
        return self._vocab_size

    def get_idx_to_char(self):
        vocab_json = self._load_character_vocab_json()

        idx_to_char = {int(k): v for k, v in vocab_json["idx_to_char"].items()}

        return idx_to_char

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
