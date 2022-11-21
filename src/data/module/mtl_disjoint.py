########################################################################################
#
# This file implement a datamodule for a MTL dataset with disjoint data
# by wrapping the speaker-only and speech-only data modules.
#
# Author(s): Nik Vaessen
########################################################################################

from dataclasses import dataclass
from typing import Optional, List

from torchdata.datapipes.iter import Zipper, Cycler

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from data_utility.eval.speaker.evaluator import SpeakerTrial

from src.data.module import (
    SpeakerRecognitionDataModule,
    SpeechRecognitionDataModule,
    SpeechRecognitionDataModuleConfig,
    SpeakerRecognitionDataModuleConfig,
)
from src.util.config_util import CastingConfig


########################################################################################
# config


@dataclass
class DisjointMTLDataModuleConfig(CastingConfig):
    speech_dm_cfg: SpeechRecognitionDataModuleConfig
    speaker_dm_cfg: SpeakerRecognitionDataModuleConfig


########################################################################################
# implementation


class DisjointMTLDataModule(LightningDataModule):
    def __init__(
        self,
        speaker_dm: SpeakerRecognitionDataModule,
        speech_dm: SpeechRecognitionDataModule,
    ):
        super(DisjointMTLDataModule, self).__init__()

        self.speaker_dm = speaker_dm
        self.speech_dm = speech_dm

    def get_idx_to_char(self):
        return self.speech_dm.get_idx_to_char()

    def get_num_train_speakers(self) -> int:
        return self.speaker_dm.get_num_train_speakers()

    def get_val_names(self):
        return ["val_speech", "val_speaker"]

    def get_val_modes(self):
        return ["speech", "speaker"]

    def get_test_speaker_eval_list(self) -> List[List[SpeakerTrial]]:
        return [
            [] for _ in self.speech_dm.get_test_names()
        ] + self.speaker_dm.get_test_speaker_eval_list()

    def get_test_names(self):
        return self.speech_dm.get_test_names() + self.speaker_dm.get_test_names()

    def get_test_modes(self):
        return ["speech" for _ in self.speech_dm.get_test_names()] + [
            "speaker" for _ in self.speaker_dm.get_test_names()
        ]

    def prepare_data(self) -> None:
        self.speaker_dm.prepare_data()
        self.speech_dm.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        self.speaker_dm.setup()
        self.speech_dm.setup()

        # train dp
        self.train_dp = Zipper(
            Cycler(self.speech_dm.train_dp), Cycler(self.speaker_dm.train_dp)
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.speech_dm.train_pipe_builder.wrap_pipe(self.train_dp)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        to_return = []

        val_speech_dl = self.speech_dm.val_dataloader()
        val_speaker_dl = self.speaker_dm.val_dataloader()

        if isinstance(val_speech_dl, list):
            to_return.extend(val_speech_dl)
        else:
            to_return.append(val_speech_dl)

        if isinstance(val_speaker_dl, list):
            to_return.extend(val_speaker_dl)
        else:
            to_return.append(val_speaker_dl)

        assert len(to_return) == 2

        return to_return

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_speech_dl = self.speech_dm.test_dataloader()
        test_speaker_dl = self.speaker_dm.test_dataloader()

        if not isinstance(test_speaker_dl, list) or not isinstance(
            test_speech_dl, list
        ):
            raise ValueError("test dataloaders should be list")

        return [] + test_speech_dl + test_speaker_dl
