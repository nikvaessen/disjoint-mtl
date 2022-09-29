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
class VoxCelebDataModuleConfig(CastingConfig):
    # path to folders containing train, val and test shards
    vox2_train: pathlib.Path
    vox2_val: pathlib.Path
    vox2_dev: pathlib.Path

    vox1_test_o: pathlib.Path
    vox1_test_e: pathlib.Path
    vox1_test_h: pathlib.Path

    # shard pattern
    shard_file_pattern: str

    # path to meta files for speaker info
    train_speaker_json: pathlib.Path

    # path to trials for speaker recognition eval
    vox2_dev_trial_path: pathlib.Path
    vox1_o_trial_path: pathlib.Path
    vox1_e_trial_path: pathlib.Path
    vox1_h_trial_path: pathlib.Path

    # which pipes to use
    task_mode: str  # 'speaker'

    # optionally, evaluate on LibriSpeech as well
    eval_on_ls: bool

    # paths to data of LibriSpeech
    ls_dev_clean_shard_path: Optional[pathlib.Path] = None
    ls_dev_other_shard_path: Optional[pathlib.Path] = None
    ls_test_clean_shard_path: Optional[pathlib.Path] = None
    ls_test_other_shard_path: Optional[pathlib.Path] = None

    # paths to trials from LibriSpeech
    ls_dev_clean_trial_path: Optional[pathlib.Path] = None
    ls_dev_other_trial_path: Optional[pathlib.Path] = None
    ls_test_clean_trial_path: Optional[pathlib.Path] = None
    ls_test_other_trial_path: Optional[pathlib.Path] = None


########################################################################################
# implementation


class VoxCelebDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: VoxCelebDataModuleConfig,
        speech_pipe_builders: Tuple[DataPipeBuilder, DataPipeBuilder, DataPipeBuilder],
        speaker_pipe_builders: Tuple[DataPipeBuilder, DataPipeBuilder, DataPipeBuilder],
    ):
        super(VoxCelebDataModule, self).__init__()

        self.cfg = cfg

        if self.cfg.task_mode == "speech":
            raise NotImplemented()

        elif self.cfg.task_mode == "speaker":
            self.train_pipe_builder = speaker_pipe_builders[0]
            self.val_pipe_builder = speaker_pipe_builders[1]
            self.test_pipe_builder = speaker_pipe_builders[2]
            self._set_speaker_to_idx()

        elif self.cfg.task_mode == "mtl":
            raise NotImplemented()

        else:
            raise ValueError(f"unknown {self.cfg.task_mode=}")

        if self.cfg.eval_on_ls:
            assert self.cfg.ls_dev_clean_shard_path is not None
            assert self.cfg.ls_dev_other_shard_path is not None
            assert self.cfg.ls_test_clean_shard_path is not None
            assert self.cfg.ls_test_other_shard_path is not None

            assert self.cfg.ls_dev_clean_trial_path is not None
            assert self.cfg.ls_dev_other_trial_path is not None
            assert self.cfg.ls_test_clean_trial_path is not None
            assert self.cfg.ls_test_other_trial_path is not None

        # init in setup()
        self.train_dp = None
        self.val_dp = None

        self.vox2_dev_dp = None
        self.vox1_o_dp = None
        self.vox1_e_dp = None
        self.vox1_h_dp = None

        self.ls_dev_dp_clean = None
        self.ls_dev_dp_other = None
        self.ls_test_dp_clean = None
        self.ls_test_dp_other = None

    def _train_speakers_json(self):
        with self.cfg.train_speaker_json.open("r") as f:
            return json.load(f)

    def _dev_speakers_json(self):
        with self.cfg.dev_speaker_json.open("r") as f:
            return json.load(f)

    def _test_speakers_json(self):
        with self.cfg.test_speaker_json.open("r") as f:
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

    def get_num_train_speakers(self) -> int:
        return len(self._train_speakers_json()["speakers"])

    def get_test_speaker_eval_list(self) -> List[List[SpeakerTrial]]:
        paths = [
            self.cfg.vox2_dev_trial_path,
            self.cfg.vox1_o_trial_path,
            self.cfg.vox1_e_trial_path,
            self.cfg.vox1_h_trial_path,
        ]

        if self.cfg.eval_on_ls:
            paths.extend(
                [
                    self.cfg.ls_dev_clean_trial_path,
                    self.cfg.ls_dev_other_trial_path,
                    self.cfg.ls_test_clean_trial_path,
                    self.cfg.ls_test_other_trial_path,
                ]
            )

        return [SpeakerTrial.from_file(f) for f in paths]

    def get_test_names(self):
        names = ["vox2_dev", "vox1_o", "vox1_e", "vox1_h"]

        if self.cfg.eval_on_ls:
            names.extend(["dev_clean", "dev_other", "test_clean", "test_other"])

        return names

    def setup(self, stage: Optional[str] = None) -> None:
        # train dp
        self.train_dp = self.train_pipe_builder.get_pipe(
            shard_dirs=[self.cfg.vox2_train],
            shard_file_pattern=self.cfg.shard_file_pattern,
        )

        # val dp
        self.val_dp = self.val_pipe_builder.get_pipe(
            shard_dirs=[self.cfg.vox2_val],
            shard_file_pattern=self.cfg.shard_file_pattern,
        )

        # test dp
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

        if self.cfg.eval_on_ls:
            self.ls_dev_dp_clean = self.test_pipe_builder.get_pipe(
                shard_dirs=self.cfg.ls_dev_clean_shard_path,
                shard_file_pattern=self.cfg.shard_file_pattern,
            )
            self.ls_dev_dp_other = self.test_pipe_builder.get_pipe(
                shard_dirs=self.cfg.ls_dev_other_shard_path,
                shard_file_pattern=self.cfg.shard_file_pattern,
            )
            self.ls_test_dp_clean = self.test_pipe_builder.get_pipe(
                shard_dirs=self.cfg.ls_test_clean_shard_path,
                shard_file_pattern=self.cfg.shard_file_pattern,
            )
            self.ls_test_dp_other = self.test_pipe_builder.get_pipe(
                shard_dirs=self.cfg.ls_test_other_shard_path,
                shard_file_pattern=self.cfg.shard_file_pattern,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_pipe_builder.wrap_pipe(self.train_dp)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_pipe_builder.wrap_pipe(self.val_dp)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dp_list = [self.vox2_dev_dp, self.vox1_o_dp, self.vox1_e_dp, self.vox1_h_dp]

        if self.cfg.eval_on_ls:
            dp_list.extend(
                [
                    self.ls_dev_dp_clean,
                    self.ls_dev_dp_other,
                    self.ls_test_dp_clean,
                    self.ls_test_dp_other,
                ]
            )

        return [self.test_pipe_builder.wrap_pipe(dp) for dp in dp_list]
