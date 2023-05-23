#! /usr/bin/env bash
set -e

# check whether paths have been set
if [[ -z "$VOXCELEB_SHARD_DIR" ]]; then
  echo "Please set VOXCELEB_SHARD_DIR before calling this script"
  exit 3
fi

if [[ -z "$VOXCELEB_META_DIR" ]]; then
  echo "Please set VOXCELEB_META_DIR before calling this script"
  exit 4
fi

# make sure all potential directories exist
mkdir -p "$VOXCELEB_SHARD_DIR" "$VOXCELEB_META_DIR"

# map each speaker to a classification index
echo "generating $VOXCELEB_META_DIR/speakers_train.json"
poetry run generate_speaker_mapping \
  "$VOXCELEB_SHARD_DIR"/vox2_train \
  "$VOXCELEB_SHARD_DIR"/vox2_val \
  --out "$VOXCELEB_META_DIR"/speakers_train.json

# create trials for speaker recognition on dev set
echo "generating $VOXCELEB_META_DIR/trials_dev.txt"
poetry run generate_speaker_trials "$VOXCELEB_SHARD_DIR"/vox2_dev* \
  --out "$VOXCELEB_META_DIR"/trials_dev.txt\
  --limit 200_000
