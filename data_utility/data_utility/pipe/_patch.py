import heapq

from functools import partial
from typing import Iterator, List, Optional, Callable

from torch.utils.data import DataChunk, IterDataPipe


########################################################################################
# keep here until new torch data release


def _default_len_fn(token):
    return len(token)


def _token_len_fn(token, len_fn):
    return len_fn(token), token


def _token_filter_fn(data, *, min_len, max_len):
    length, token = data
    return length >= min_len and length <= max_len


class MaxTokenBucketizerIterDataPipe:
    r"""
    Creates mini-batches of data from a min-heap with limited size, and the total length of samples
    returned by ``len_fn`` within each batch will be limited by ``max_token_count``
    (functional name: ``max_token_bucketize``). If ``min_len`` or ``max_len`` is set, the samples with
    length that is out of ``[min_len, max_len]`` will be filtered out.

    The purpose of this DataPipe is to batch samples with similar length according to ``len_fn``.
    Min-heap is used here to make sure the samples are sorted incrementally based on the length. And,
    the total length of samples in each batch is guaranteed to be smaller than ``max_token_count``.
    For an example in the audio domain, it may be batching samples with similar length. Then, given the
    ``max_token_count``, each batch may be concatenated to a Tensor with the same size and minimum padding.

    If ``padded_tokens`` is set to `True`, a batch of samples will never exceed the given ``max_token_count``,
    even if they are padded to equal length.

    Args:
        datapipe: Iterable DataPipe being batched
        max_token_count: Maximum length of total length of data in each batch
        len_fn: Function to be applied to each element to get lengths. ``len(data)`` is used by default.
        min_len: Optional minimum length to be included into each batch
        max_len: Optional maximum length to be included into each batch.
        buffer_size: This restricts how many tokens are taken from prior DataPipe to bucketize
        padded_tokens: If true, assume data will be padded to the largest length in bucket.


    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(['1', '11', '1', '1111', '111', '1', '11', '11', '111'])
        >>> # Using default len_fn to sort samples based on length (string length in this case)
        >>> batch_dp = source_dp.max_token_bucketize(max_token_count=5)
        >>> list(batch_dp)
        [['1', '1', '1', '11'], ['11', '11'], ['111'], ['111'], ['1111']]
        >>> batch_dp = source_dp.max_token_bucketize(max_token_count=4, buffer_size=4)
        >>> list(batch_dp)
        [['1', '1', '1'], ['11', '11'], ['11'], ['111'], ['111'], ['1111']]
    """
    datapipe: IterDataPipe
    max_token_count: int
    len_fn: Callable
    min_len: int
    max_len: Optional[int]
    buffer_size: int

    def __init__(
        self,
        datapipe: IterDataPipe,
        max_token_count: int,
        len_fn: Callable = _default_len_fn,
        min_len: int = 0,
        max_len: Optional[int] = None,
        buffer_size: int = 1000,
        padded_tokens: bool = False,
    ) -> None:
        if max_len is None:
            max_len = max_token_count

        if min_len < 0 or min_len > max_len:
            raise ValueError(
                "``min_len`` should be larger than 0 and equal to or smaller than ``max_len``."
            )
        if max_len > max_token_count:
            raise ValueError(
                "``max_token_count`` must be equal to or greater than ``max_len``."
            )
        datapipe = datapipe.map(partial(_token_len_fn, len_fn=len_fn))
        datapipe = datapipe.filter(
            partial(_token_filter_fn, min_len=min_len, max_len=max_len)
        )
        if buffer_size <= 0:
            raise ValueError("'buffer_size' is required to be a positive integer.")
        self.datapipe = datapipe
        self.max_token_count = max_token_count
        self.buffer_size = buffer_size
        self.padded_tokens = padded_tokens

    def __iter__(self) -> Iterator[DataChunk]:
        buffer: List = []
        batch: List = []
        batch_size: int = 0
        max_length: int = 0
        for d in self.datapipe:
            heapq.heappush(buffer, d)
            if len(buffer) == self.buffer_size:
                length, token = heapq.heappop(buffer)
                max_length = max(length, max_length)
                if self.padded_tokens:
                    new_batch_size = (len(batch) + 1) * max_length
                else:
                    new_batch_size = batch_size + length
                if new_batch_size > self.max_token_count:
                    yield DataChunk(batch)
                    batch = [token]
                    batch_size = length
                    max_length = length
                else:
                    batch.append(token)
                    batch_size = new_batch_size
        while buffer:
            length, token = heapq.heappop(buffer)
            max_length = max(length, max_length)
            if self.padded_tokens:
                new_batch_size = (len(batch) + 1) * max_length
            else:
                new_batch_size = batch_size + length
            if new_batch_size > self.max_token_count:
                yield DataChunk(batch)
                batch = [token]
                batch_size = length
                max_length = length
            else:
                batch.append(token)
                batch_size = new_batch_size
        if batch:
            yield DataChunk(batch)
