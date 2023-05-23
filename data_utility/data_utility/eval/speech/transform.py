########################################################################################
#
# Logic to encode a transcription to indexes, and decode network predictions to
# transcriptions.
#
# Author(s): Nik Vaessen
########################################################################################

import json
import pathlib

from typing import Dict, Optional, List, FrozenSet

import torch as t
import tempfile

from transformers.models.wav2vec2.tokenization_wav2vec2 import Wav2Vec2CTCTokenizer

########################################################################################
# off-load actual encoding/decoding to transformers library

_tokenizer_map: Dict[FrozenSet, Wav2Vec2CTCTokenizer] = {}


BLANK_TOKEN = "<blank>"
BLANK_TOKEN_IDX = 0

BEGIN_SENTENCE_TOKEN = "<s>"
BEGIN_SENTENCE_TOKEN_IDX = 1

END_SENTENCE_TOKEN = "</s>"
END_SENTENCE_TOKEN_IDX = 2

UNKNOWN_TOKEN = "<unk>"
UNKNOWN_TOKEN_IDX = 3

WORD_DELIM_TOKEN = "|"
WORD_DELIM_TOKEN_IDX = 4


def get_default_token_list():
    dictionary = {
        BLANK_TOKEN: BLANK_TOKEN_IDX,
        BEGIN_SENTENCE_TOKEN: BEGIN_SENTENCE_TOKEN_IDX,
        END_SENTENCE_TOKEN: END_SENTENCE_TOKEN_IDX,
        UNKNOWN_TOKEN: UNKNOWN_TOKEN_IDX,
        WORD_DELIM_TOKEN: WORD_DELIM_TOKEN_IDX,
    }

    return [k for k, v in sorted(dictionary.items(), key=lambda tup: tup[1])]


def _get_tokenizer(idx_to_char: Dict[int, str]):
    set_of_dict = frozenset([(k, v) for k, v in idx_to_char.items()])

    if set_of_dict in _tokenizer_map:
        return _tokenizer_map[set_of_dict]
    else:
        with tempfile.TemporaryDirectory() as directory:
            # write int_to_char to a (temporary) file
            vocab_path = pathlib.Path(directory) / "vocab.json"
            with vocab_path.open("w") as f:
                json.dump({v: k for k, v in idx_to_char.items()}, f)

            assert idx_to_char[BLANK_TOKEN_IDX] == BLANK_TOKEN
            assert idx_to_char[BEGIN_SENTENCE_TOKEN_IDX] == BEGIN_SENTENCE_TOKEN
            assert idx_to_char[END_SENTENCE_TOKEN_IDX] == END_SENTENCE_TOKEN
            assert idx_to_char[UNKNOWN_TOKEN_IDX] == UNKNOWN_TOKEN
            assert idx_to_char[WORD_DELIM_TOKEN_IDX] == WORD_DELIM_TOKEN

            tokenizer = Wav2Vec2CTCTokenizer(
                str(vocab_path),
                bos_token=BEGIN_SENTENCE_TOKEN,
                eos_token=END_SENTENCE_TOKEN,
                unk_token=UNKNOWN_TOKEN,
                pad_token=BLANK_TOKEN,
                word_delimiter_token=WORD_DELIM_TOKEN,
            )

        _tokenizer_map[set_of_dict] = tokenizer
        return tokenizer


########################################################################################
# encode


def encode_transcription(transcription: str, char_to_idx: Dict[str, int]) -> t.Tensor:
    tokenizer = _get_tokenizer({v: k for k, v in char_to_idx.items()})
    idx_list = tokenizer(text_target=transcription)["input_ids"]

    return t.tensor(idx_list, dtype=t.long)


########################################################################################
# decode


def decode_predictions_greedy(
    predictions: t.Tensor,
    idx_to_char: Dict[int, str],
    until_seq_idx: Optional[List[int]] = None,
):
    # assume predictions are shape [BATCH_SIZE, SEQ_LENGTH, NUM_CHARS]
    assert len(predictions.shape) == 3
    assert len(until_seq_idx) == predictions.shape[0]

    greedy_prediction = t.argmax(predictions, dim=2)

    transcriptions = []
    for i in range(greedy_prediction.shape[0]):
        idx_sequence = greedy_prediction[i, :].squeeze()
        transcriptions.append(
            decode_idx_sequence(
                idx_sequence,
                idx_to_char,
                until_seq_idx[i] if until_seq_idx is not None else None,
            )
        )

    return transcriptions


def decode_idx_sequence(
    idx_sequence: t.Tensor, idx_to_char: Dict[int, str], until_idx: Optional[int] = None
):
    assert idx_to_char[0] == "<blank>"

    # assume idx_sequence is a vector of integers
    assert len(idx_sequence.shape) == 1

    idx_sequence = idx_sequence.tolist()

    if until_idx is not None:
        idx_sequence = idx_sequence[:until_idx]

    tokenizer = _get_tokenizer(idx_to_char)

    return tokenizer.decode(idx_sequence)
