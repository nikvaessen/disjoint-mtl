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

# make the CSV file for each of the 3 train splits
sw_generate_csv \
--dir "$SWITCHBOARD_EXTRACT_DIR"/swb1/utterance \
--csv "$SWITCHBOARD_SHARD_DIR"/test/_meta.test.csv \
--speakers "$SWITCHBOARD_EXTRACT_DIR"/swb1/tables/tables/caller.tab \
--ext wav
