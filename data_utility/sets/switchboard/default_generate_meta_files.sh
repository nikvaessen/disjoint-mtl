#! /usr/bin/env bash
set -e

# check whether paths have been set
if [[ -z "$SWITCHBOARD_SHARD_DIR" ]]; then
  echo "Please set SWITCHBOARD_SHARD_DIR before calling this script"
  exit 3
fi

if [[ -z "$SWITCHBOARD_META_DIR" ]]; then
  echo "Please set SWITCHBOARD_META_DIR before calling this script"
  exit 4
fi

# make sure all potential directories exist
mkdir -p "$SWITCHBOARD_SHARD_DIR" "$SWITCHBOARD_META_DIR"

# map each character to a classification index
echo "generating $SWITCHBOARD_META_DIR/character_vocabulary.json"
poetry run generate_character_vocabulary \
  "$SWITCHBOARD_SHARD_DIR"/test \
  --out "$SWITCHBOARD_META_DIR"/character_vocabulary.json


# create trials for speaker recognition on val, dev and test set
echo "generating $SWITCHBOARD_META_DIR/trials_eval.txt"
poetry run generate_speaker_trials \
  "$SWITCHBOARD_SHARD_DIR"/test \
 --out "$SWITCHBOARD_META_DIR"/trials_eval.txt \
 --same-sex true \
 --session-overlap false \
 --limit 1_000_000