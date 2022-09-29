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
class LibriSpeechDataModuleConfig(CastingConfig):
    # path to folders containing train, val and test shards
    train_c100_shard_path: pathlib.Path
    train_c360_shard_path: pathlib.Path
    train_o500_shard_path: pathlib.Path

    val_c100_shard_path: pathlib.Path
    val_c360_shard_path: pathlib.Path
    val_o500_shard_path: pathlib.Path

    dev_clean_shard_path: pathlib.Path
    dev_other_shard_path: pathlib.Path

    test_clean_shard_path: pathlib.Path
    test_other_shard_path: pathlib.Path

    # shard pattern
    shard_file_pattern: str

    # path to meta files for speaker info
    train_speaker_json: pathlib.Path

    # path to meta file for speech info
    char_vocab_json: pathlib.Path

    # path to trials for speaker recognition eval
    dev_clean_trial_path: pathlib.Path
    dev_other_trial_path: pathlib.Path
    test_clean_trial_path: pathlib.Path
    test_other_trial_path: pathlib.Path

    # train set options
    # 960h: clean-100, clean-360 and other-500 subsets
    # 100h: clean-100 subset
    train_set_mode: str

    # which pipes to use
    task_mode: str  # 'speech', 'speaker' or 'mtl'

    # optionally, evaluate on VoxCeleb as well
    eval_on_voxceleb: bool

    # paths to data of LibriSpeech
    vox2_dev: Optional[pathlib.Path] = None
    vox1_test_o: Optional[pathlib.Path] = None
    vox1_test_e: Optional[pathlib.Path] = None
    vox1_test_h: Optional[pathlib.Path] = None

    # path to trials for speaker recognition eval
    vox2_dev_trial_path: Optional[pathlib.Path] = None
    vox1_o_trial_path: Optional[pathlib.Path] = None
    vox1_e_trial_path: Optional[pathlib.Path] = None
    vox1_h_trial_path: Optional[pathlib.Path] = None


########################################################################################
# implementation


class LibriSpeechDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: LibriSpeechDataModuleConfig,
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

        self.vox2_dev_dp = None
        self.vox1_o_dp = None
        self.vox1_e_dp = None
        self.vox1_h_dp = None

    def _train_speakers_json(self):
        with self.cfg.train_speaker_json.open("r") as f:
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
        paths = [
            self.cfg.dev_clean_trial_path,
            self.cfg.dev_other_trial_path,
            self.cfg.test_clean_trial_path,
            self.cfg.test_other_trial_path,
        ]

        if self.cfg.task_mode == "speaker" and self.cfg.eval_on_voxceleb:
            paths.extend(
                [
                    self.cfg.vox2_dev_trial_path,
                    self.cfg.vox1_o_trial_path,
                    self.cfg.vox1_e_trial_path,
                    self.cfg.vox1_h_trial_path,
                ]
            )

        return [SpeakerTrial.from_file(f) for f in []]

    def get_test_names(self):
        names = ["dev_clean", "dev_other", "test_clean", "test_other"]

        if self.cfg.task_mode == "speaker" and self.cfg.eval_on_voxceleb:
            names.extend(["vox2_dev", "vox1_o", "vox1_e", "vox1_h"])

    def get_vocab_size(self):
        return len(self._character_vocab_json()["characters"])

    def get_idx_to_char(self):
        vocab_json = self._character_vocab_json()

        idx_to_char = {int(k): v for k, v in vocab_json["idx_to_char"].items()}

        return idx_to_char

    def setup(self, stage: Optional[str] = None) -> None:
        # train dp
        if self.cfg.train_set_mode == "960h":
            self.train_dp = self.train_pipe_builder.get_pipe(
                shard_dirs=[
                    self.cfg.train_c100_shard_path,
                    self.cfg.train_c360_shard_path,
                    self.cfg.train_o500_shard_path,
                ],
                shard_file_pattern=self.cfg.shard_file_pattern,
            )
        elif self.cfg.train_set_mode == "100h":
            self.train_dp = self.train_pipe_builder.get_pipe(
                shard_dirs=self.cfg.train_c100_shard_path,
                shard_file_pattern=self.cfg.shard_file_pattern,
            )
        else:
            raise ValueError(f"unknown {self.cfg.train_set_mode=}")

        # val dp
        self.val_dp = self.val_pipe_builder.get_pipe(
            shard_dirs=[
                self.cfg.val_c100_shard_path,
                self.cfg.val_c360_shard_path,
                self.cfg.val_o500_shard_path,
            ],
            shard_file_pattern=self.cfg.shard_file_pattern,
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

        if self.cfg.task_mode == "speaker" and self.cfg.eval_on_voxceleb:
            self.vox2_dev_dp = self.test_pipe_builder.get_pipe(
                shard_dirs=self.cfg.vox2_dev,
                shard_file_pattern=self.cfg.shard_file_pattern,
            )
            self.vox1_o_dp = self.test_pipe_builder.get_pipe(
                shard_dirs=self.cfg.vox1_test_o,
                shard_file_pattern=self.cfg.shard_file_pattern,
            )
            self.vox1_e_dp = self.test_pipe_builder.get_pipe(
                shard_dirs=self.cfg.vox1_test_e,
                shard_file_pattern=self.cfg.shard_file_pattern,
            )
            self.vox1_h_dp = self.test_pipe_builder.get_pipe(
                shard_dirs=self.cfg.vox1_test_h,
                shard_file_pattern=self.cfg.shard_file_pattern,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_pipe_builder.wrap_pipe(self.train_dp)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_pipe_builder.wrap_pipe(self.val_dp)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dp_list = [
            self.dev_dp_clean,
            self.dev_dp_other,
            self.test_dp_clean,
            self.test_dp_other,
        ]

        if self.cfg.task_mode == "speaker" and self.cfg.eval_on_voxceleb:
            dp_list.extend(
                [self.vox2_dev_dp, self.vox1_o_dp, self.vox1_e_dp, self.vox1_h_dp]
            )

        return [self.test_pipe_builder.wrap_pipe(dp) for dp in dp_list]
