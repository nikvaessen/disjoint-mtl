#! /usr/bin/env bash
set -e

# check whether paths have been set
if [[ -z "$LIBRISPEECH_SHARD_DIR" ]]; then
  echo "Please set LIBRISPEECH_SHARD_DIR before calling this script"
  exit 3
fi

if [[ -z "$LIBRISPEECH_META_DIR" ]]; then
  echo "Please set LIBRISPEECH_META_DIR before calling this script"
  exit 4
fi

# make sure all potential directories exist
mkdir -p "$LIBRISPEECH_SHARD_DIR" "$LIBRISPEECH_META_DIR"

# map each character to a classification index
echo "generating $LIBRISPEECH_META_DIR/character_vocabulary.json"
poetry run generate_character_vocabulary \
  "$LIBRISPEECH_SHARD_DIR"/train-clean-100 \
  "$LIBRISPEECH_SHARD_DIR"/train-clean-360 \
  "$LIBRISPEECH_SHARD_DIR"/train-other-500 \
  "$LIBRISPEECH_SHARD_DIR"/val-clean-100 \
  "$LIBRISPEECH_SHARD_DIR"/val-clean-360 \
  "$LIBRISPEECH_SHARD_DIR"/val-other-500 \
  "$LIBRISPEECH_SHARD_DIR"/dev-clean \
  "$LIBRISPEECH_SHARD_DIR"/dev-other \
  "$LIBRISPEECH_SHARD_DIR"/test-clean \
  "$LIBRISPEECH_SHARD_DIR"/test-other \
  --out "$LIBRISPEECH_META_DIR"/character_vocabulary.json

echo "generating $LIBRISPEECH_META_DIR/character_distribution_train.json"
poetry run generate_character_distribution \
  "$LIBRISPEECH_SHARD_DIR"/train-clean-100 \
  "$LIBRISPEECH_SHARD_DIR"/train-clean-360 \
  "$LIBRISPEECH_SHARD_DIR"/train-other-500 \
  "$LIBRISPEECH_SHARD_DIR"/val-clean-100 \
  "$LIBRISPEECH_SHARD_DIR"/val-clean-360 \
  "$LIBRISPEECH_SHARD_DIR"/val-other-500 \
  --out "$LIBRISPEECH_META_DIR"/character_distribution_train.json

# map each speaker to a classification index
echo "generating $LIBRISPEECH_META_DIR/speakers_train.json"
poetry run generate_speaker_mapping \
  "$LIBRISPEECH_SHARD_DIR"/train-clean-100 \
  "$LIBRISPEECH_SHARD_DIR"/train-clean-360 \
  "$LIBRISPEECH_SHARD_DIR"/train-other-500 \
  "$LIBRISPEECH_SHARD_DIR"/val-clean-100 \
  "$LIBRISPEECH_SHARD_DIR"/val-clean-360 \
  "$LIBRISPEECH_SHARD_DIR"/val-other-500 \
  --out "$LIBRISPEECH_META_DIR"/speakers_train.json

echo "generating $LIBRISPEECH_META_DIR/speakers_train_clean.json"
poetry run generate_speaker_mapping \
  "$LIBRISPEECH_SHARD_DIR"/train-clean-100 \
  "$LIBRISPEECH_SHARD_DIR"/train-clean-360 \
  "$LIBRISPEECH_SHARD_DIR"/val-clean-100 \
  "$LIBRISPEECH_SHARD_DIR"/val-clean-360 \
  --out "$LIBRISPEECH_META_DIR"/speakers_train_clean.json

echo "generating $LIBRISPEECH_META_DIR/speakers_train_other.json"
poetry run generate_speaker_mapping \
  "$LIBRISPEECH_SHARD_DIR"/train-other-500 \
  "$LIBRISPEECH_SHARD_DIR"/val-other-500 \
  --out "$LIBRISPEECH_META_DIR"/speakers_train_other.json

# create trials for speaker recognition on val, dev and test set
echo "generating $LIBRISPEECH_META_DIR/trials_val.txt"
poetry run generate_speaker_trials "$LIBRISPEECH_SHARD_DIR"/val-* --out "$LIBRISPEECH_META_DIR"/trials_val.txt
echo "generating $LIBRISPEECH_META_DIR/trials_dev_clean.txt"
poetry run generate_speaker_trials "$LIBRISPEECH_SHARD_DIR"/dev-clean  --out "$LIBRISPEECH_META_DIR"/trials_dev_clean.txt
echo "generating $LIBRISPEECH_META_DIR/trials_dev_other.txt"
poetry run generate_speaker_trials "$LIBRISPEECH_SHARD_DIR"/dev-other  --out "$LIBRISPEECH_META_DIR"/trials_dev_other.txt
echo "generating $LIBRISPEECH_META_DIR/trials_test_clean.txt"
poetry run generate_speaker_trials "$LIBRISPEECH_SHARD_DIR"/test-clean --out "$LIBRISPEECH_META_DIR"/trials_test_clean.txt
echo "generating $LIBRISPEECH_META_DIR/trials_test_other.txt"
poetry run generate_speaker_trials "$LIBRISPEECH_SHARD_DIR"/test-other --out "$LIBRISPEECH_META_DIR"/trials_test_other.txt
