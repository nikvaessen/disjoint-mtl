#! /usr/bin/env bash
set -e

# check all environment variables
if [[ -z "$SWITCHBOARD_EXTRACT_DIR" ]]; then
  echo "Please set SWITCHBOARD_EXTRACT_DIR before calling this script"
  exit 2
fi

if [[ -z "$SWITCHBOARD_SHARD_DIR" ]]; then
  echo "Please set SWITCHBOARD_SHARD_DIR before calling this script"
  exit 3
fi

# run script to generate utterances out of the long data recordings
# of switchboard to mimic directory style of librispeech and voxceleb
poetry run sw_to_utterance \
  --dir "$SWITCHBOARD_EXTRACT_DIR"/swb1 \
  --out "$SWITCHBOARD_EXTRACT_DIR"/swb1/utterance \
  --min 3 \
  --max 20 \
  --ses 2