#! /usr/bin/env bash
set -e

# check whether paths have been set
if [[ -z "$LIBRISPEECH_EXTRACT_DIR" ]]; then
  echo "Please set LIBRISPEECH_EXTRACT_DIR before calling this script"
  exit 2
fi

if [[ -z "$LIBRISPEECH_SHARD_DIR" ]]; then
  echo "Please set LIBRISPEECH_SHARD_DIR before calling this script"
  exit 3
fi

if [[ -z "$LIBRISPEECH_META_DIR" ]]; then
  echo "Please set LIBRISPEECH_META_DIR before calling this script"
  exit 4
fi

# make sure all potential directories exist
mkdir -p "$LIBRISPEECH_EXTRACT_DIR" "$LIBRISPEECH_SHARD_DIR" "$LIBRISPEECH_META_DIR"

# folders for each split
train_clean_100=$LIBRISPEECH_SHARD_DIR/train-clean-100
train_clean_360=$LIBRISPEECH_SHARD_DIR/train-clean-360
train_other_500=$LIBRISPEECH_SHARD_DIR/train-other-500

val_clean_100=$LIBRISPEECH_SHARD_DIR/val-clean-100
val_clean_360=$LIBRISPEECH_SHARD_DIR/val-clean-360
val_other_500=$LIBRISPEECH_SHARD_DIR/val-other-500

dev_clean=$LIBRISPEECH_SHARD_DIR/dev-clean
dev_other=$LIBRISPEECH_SHARD_DIR/dev-other

test_clean=$LIBRISPEECH_SHARD_DIR/test-clean
test_other=$LIBRISPEECH_SHARD_DIR/test-other

# make folders
mkdir -p "$train_clean_100" "$train_clean_360" "$train_other_500"
mkdir -p "$val_clean_100" "$val_clean_360" "$val_other_500"
mkdir -p "$dev_clean" "$dev_other"
mkdir -p "$test_clean" "$test_other"

# make the CSV file for each of the 3 train splits
ls_generate_csv \
--dir "$LIBRISPEECH_EXTRACT_DIR"/LibriSpeech/train-clean-100 \
--csv "$train_clean_100"/_meta.train_clean_100.csv \
--speakers "$LIBRISPEECH_META_DIR"/_archive_speakers.txt \
--ext wav

ls_generate_csv \
--dir "$LIBRISPEECH_EXTRACT_DIR"/LibriSpeech/train-clean-360 \
--csv "$train_clean_360"/_meta.train_clean_360.csv \
--speakers "$LIBRISPEECH_META_DIR"/_archive_speakers.txt \
--ext wav

ls_generate_csv \
--dir "$LIBRISPEECH_EXTRACT_DIR"/LibriSpeech/train-other-500 \
--csv "$train_other_500"/_meta.train_other_500.csv \
--speakers "$LIBRISPEECH_META_DIR"/_archive_speakers.txt \
--ext wav

# split the train splits into train/val
VAL_RATIO=0.05
split_csv "$train_clean_100"/_meta.train_clean_100.csv  \
--delete-in \
--strategy by_recordings --ratio $VAL_RATIO \
--remain-out "$train_clean_100"/_meta.train_clean_100.csv \
--split-out "$val_clean_100"/_meta.val_clean_100.csv \

split_csv "$train_clean_360"/_meta.train_clean_360.csv  \
--delete-in \
--strategy by_recordings --ratio $VAL_RATIO \
--remain-out "$train_clean_360"/_meta.train_clean_360.csv \
--split-out "$val_clean_360"/_meta.val_clean_360.csv \

split_csv "$train_other_500"/_meta.train_other_500.csv \
--delete-in \
--strategy by_recordings --ratio $VAL_RATIO \
--remain-out "$train_other_500"/_meta.train_other_500.csv \
--split-out "$val_other_500"/_meta.val_other_500.csv \

# make the CSV file for each dev split split
ls_generate_csv \
--dir "$LIBRISPEECH_EXTRACT_DIR"/LibriSpeech/dev-clean \
--csv "$dev_clean"/_meta.dev_clean.csv \
--speakers "$LIBRISPEECH_META_DIR"/_archive_speakers.txt \
--ext wav

ls_generate_csv \
--dir "$LIBRISPEECH_EXTRACT_DIR"/LibriSpeech/dev-other \
--csv "$dev_other"/_meta.dev_other.csv \
--speakers "$LIBRISPEECH_META_DIR"/_archive_speakers.txt \
--ext wav


# make the CSV file for each test split split
ls_generate_csv \
--dir "$LIBRISPEECH_EXTRACT_DIR"/LibriSpeech/test-clean \
--csv "$test_clean"/_meta.test_clean.csv \
--speakers "$LIBRISPEECH_META_DIR"/_archive_speakers.txt \
--ext wav

ls_generate_csv \
--dir "$LIBRISPEECH_EXTRACT_DIR"/LibriSpeech/test-other \
--csv "$test_other"/_meta.test_other.csv \
--speakers "$LIBRISPEECH_META_DIR"/_archive_speakers.txt \
--ext wav

