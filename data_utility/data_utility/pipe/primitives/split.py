########################################################################################
#
# Split a datapipe of `WavAudioDataSample` to multiple downstream datapipes.
#
# Author(s): Nik Vaessen
########################################################################################

from typing import List, Set

from torchdata.datapipes.iter import IterDataPipe, Demultiplexer

from data_utility.pipe.containers import WavAudioDataSample


########################################################################################
# Split a pipeline into multiple downstream pipelines


def split_on_speaker_id(
    dp: IterDataPipe[WavAudioDataSample], splits: List[Set[str]]
) -> List[IterDataPipe[WavAudioDataSample]]:
    classification_map = {}

    for idx, collection in enumerate(splits):
        for speaker_id in collection:
            classification_map[speaker_id] = int(idx)

    def classification_fn(element):
        assert isinstance(element, WavAudioDataSample)

        if element.speaker_id in classification_map:
            return classification_map[element.speaker_id]
        else:
            return None

    return Demultiplexer(
        dp,
        num_instances=len(splits),
        classifier_fn=classification_fn,
    )
