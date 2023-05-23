########################################################################################
#
# Implement a datapipe which can chunk audio into small(er) pieces
#
# Author(s): Nik Vaessen
########################################################################################

from copy import deepcopy

import numpy as np

from torchdata.datapipes.iter import IterDataPipe

from data_utility.pipe.containers import WavAudioDataSample


########################################################################################
# Chunk the audio into smaller pieces


class ChunkAudioSampleIterDataPipe(IterDataPipe):
    known_chunk_strategy = ["none", "random", "start", "contiguous"]

    def __init__(
        self,
        source_dp: IterDataPipe,
        chunk_strategy: str,
        chunk_size_seconds: float,
        sample_rate: int,
    ) -> None:
        super().__init__()
        self.source_dp = source_dp
        self.chunk_strategy = chunk_strategy

        if self.chunk_strategy not in self.known_chunk_strategy:
            raise ValueError(
                f"{chunk_strategy=} not in {ChunkAudioSampleIterDataPipe.known_chunk_strategy=}"
            )

        if self.chunk_strategy != "none":
            self.chunk_num_frames = self.compute_chunk_size_frames(
                chunk_size_seconds, sample_rate
            )

    def __iter__(self):
        if self.chunk_strategy == "start":
            return self.strategy_chunk_from_beginning()
        elif self.chunk_strategy == "random":
            return self.strategy_chunk_random()
        elif self.chunk_strategy == "contiguous":
            return self.strategy_chunk_contiguous()
        elif self.chunk_strategy == "none":
            return iter(self.source_dp)
        else:
            raise ValueError(f"unknown {self.chunk_strategy=}")

    @staticmethod
    def compute_chunk_size_frames(seconds: float, sample_rate: int = 16_000):
        return round(seconds * sample_rate)

    def strategy_chunk_from_beginning(self):
        for element in self.source_dp:
            assert isinstance(element, WavAudioDataSample)
            element.audio_tensor = element.audio_tensor[
                ..., : self.chunk_num_frames
            ].clone()
            element.audio_length_frames = element.audio_tensor.shape[1]
            yield element

    def strategy_chunk_random(self):
        for element in self.source_dp:
            assert isinstance(element, WavAudioDataSample)
            num_samples = element.audio_tensor.shape[1]

            if self.chunk_num_frames >= num_samples:
                audio_tensor = element.audio_tensor[..., :]
            else:
                start = np.random.randint(
                    low=0, high=num_samples - self.chunk_num_frames + 1
                )
                end = start + self.chunk_num_frames

                audio_tensor = element.audio_tensor[..., start:end]
                assert audio_tensor.shape[1] == self.chunk_num_frames

            element.audio_tensor = audio_tensor.clone()
            element.audio_length_frames = element.audio_tensor.shape[1]

            yield element

    def strategy_chunk_contiguous(self):
        for element in self.source_dp:
            assert isinstance(element, WavAudioDataSample)

            num_samples = element.audio_tensor.shape[1]
            num_possible_chunks = num_samples // self.chunk_num_frames

            for selected_chunk in range(num_possible_chunks):
                start = selected_chunk * self.chunk_num_frames
                end = start + self.chunk_num_frames

                sample_copy = deepcopy(element)
                sample_copy.audio_tensor = sample_copy.audio_tensor[start:end].clone()
                sample_copy.audio_length_frames = sample_copy.audio_tensor.shape[1]

                yield sample_copy
