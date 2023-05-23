################################################################################
#
# Implement an Evaluator object which encapsulates the process
# computing performance metric of speech recognition task.
#
# Author(s): Nik Vaessen
################################################################################

import pathlib

from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple
from warnings import warn

import numpy as np
import torch as t
import pandas as pd

from torch.nn.functional import normalize

from data_utility.eval.speaker.speaker_eval_metrics import calculate_eer, calculate_mdc


################################################################################
# define data structures required for evaluating


@dataclass
class EmbeddingSample:
    sample_id: str
    embedding: t.Tensor


########################################################################################
# Container encapsulating a speaker trial


@dataclass
class SpeakerTrial:
    left: str
    right: str
    same_speaker: bool

    def __eq__(self, other):
        if isinstance(other, SpeakerTrial):
            return self.__hash__() == other.__hash__()

        return False

    def __hash__(self):
        return frozenset([self.left, self.right, self.same_speaker]).__hash__()

    def __str__(self):
        assert self.left.count(" ") == 0
        assert self.right.count(" ") == 0

        bool_str = str(int(self.same_speaker))
        return f"{self.left} {self.right} {bool_str}"

    def to_line(self):
        return str(self)

    @classmethod
    def from_line(cls, line: str):
        assert line.count(" ") == 2

        left, right, bool_str = line.strip().split(" ")
        bool_value = bool(int(bool_str))

        assert len(left) > 0
        assert len(right) > 0

        return SpeakerTrial(left=left, right=right, same_speaker=bool_value)

    @classmethod
    def to_file(cls, file_path: pathlib.Path, trials: List["SpeakerTrial"]):
        with file_path.open("w") as f:
            f.writelines([f"{tr.to_line()}\n" for tr in trials])

    @classmethod
    def from_file(cls, file_path: pathlib.Path) -> List["SpeakerTrial"]:
        with file_path.open("r") as f:
            return [SpeakerTrial.from_line(s) for s in f.readlines()]


################################################################################
# abstract base class of an evaluator


class SpeakerRecognitionEvaluator:
    def evaluate(
        self,
        pairs: List[SpeakerTrial],
        samples: List[EmbeddingSample],
        print_info: bool = False,
    ) -> Dict[str, float]:
        # create a hashmap for quicker access to samples based on key
        sample_map = {}

        for sample in samples:
            if sample.sample_id in sample_map:
                raise ValueError(f"duplicate key {sample.sample_id}")

            sample_map[sample.sample_id] = sample

        # compute a list of ground truth scores and prediction scores
        ground_truth_scores = []
        prediction_pairs = []

        for pair in pairs:
            if pair.left not in sample_map or pair.right not in sample_map:
                warn(f"{pair.left=} or {pair.right=} not in sample_map")
                return {
                    "eer": -1,
                    "eer_threshold": -1,
                    "mdc": -1,
                    "mdc_threshold": -1,
                }

            s1 = sample_map[pair.left]
            s2 = sample_map[pair.right]

            gt = 1 if pair.same_speaker else 0

            ground_truth_scores.append(gt)
            prediction_pairs.append((s1, s2))

        if len(prediction_pairs) > 100_000:
            prediction_scores = []

            for i in range(0, len(prediction_pairs), 100_000):
                sublist = prediction_pairs[i : i + 100_000]
                prediction_scores.extend(self._compute_prediction_scores(sublist))
        else:
            prediction_scores = self._compute_prediction_scores(prediction_pairs)

        # normalize scores to be between 0 and 1
        prediction_scores = np.clip((np.array(prediction_scores) + 1) / 2, 0, 1)
        prediction_scores = prediction_scores.tolist()

        # info statistics on ground-truth and prediction scores
        if print_info:
            print("ground truth scores")
            print(pd.DataFrame(ground_truth_scores).describe())
            print("prediction scores")
            print(pd.DataFrame(prediction_scores).describe())

        # compute EER
        try:
            eer, eer_threshold = calculate_eer(
                ground_truth_scores, prediction_scores, pos_label=1
            )
        except (ValueError, ZeroDivisionError) as e:
            # if NaN values, we just return a very bad score
            # so that hparam searches don't crash
            print(f"EER calculation had {e}")
            eer = 1
            eer_threshold = 1337

        # compute mdc
        try:
            mdc, mdc_threshold = calculate_mdc(ground_truth_scores, prediction_scores)
        except (ValueError, ZeroDivisionError) as e:
            print(f"mdc calculation had {e}")
            mdc = 1
            mdc_threshold = 1337

        return {
            "eer": float(eer),
            "eer_threshold": float(eer_threshold),
            "mdc": float(mdc),
            "mdc_threshold": float(mdc_threshold),
        }

    @abstractmethod
    def _compute_prediction_scores(
        self, pairs: List[Tuple[EmbeddingSample, EmbeddingSample]]
    ) -> List[float]:
        pass

    def _transform_pairs_to_tensor(
        self, pairs: List[Tuple[EmbeddingSample, EmbeddingSample]]
    ):
        # construct the comparison batches
        b1 = []
        b2 = []

        for s1, s2 in pairs:
            b1.append(s1.embedding)
            b2.append(s2.embedding)

        b1 = t.stack(b1)
        b2 = t.stack(b2)

        return b1, b2

    @abstractmethod
    def fit_parameters(
        self, embedding_tensors: List[t.Tensor], label_tensors: List[t.Tensor]
    ):
        pass

    @abstractmethod
    def reset_parameters(self):
        pass


################################################################################
# Utility methods common between evaluators


def compute_mean_std_batch(all_tensors: t.Tensor):
    # compute mean and std over each dimension of EMBEDDING_SIZE
    # with a tensor of shape [NUM_SAMPLES, EMBEDDING_SIZE]
    std, mean = t.std_mean(all_tensors, dim=0)

    return mean, std


def center_batch(embedding_tensor: t.Tensor, mean: t.Tensor, std: t.Tensor):
    # center the batch with shape [NUM_PAIRS, EMBEDDING_SIZE]
    # using the computed mean and std
    centered = (embedding_tensor - mean) / (std + 1e-12)

    return centered


def length_norm_batch(embedding_tensor: t.Tensor):
    # length normalize the batch with shape [NUM_PAIRS, EMBEDDING_SIZE]
    return normalize(embedding_tensor, dim=1)
