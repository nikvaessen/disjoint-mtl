#! /usr/bin/env bash
set -e

# check all environment variables
if [[ -z "$SWITCHBOARD_RAW_DIR" ]]; then
  echo "Please set SWITCHBOARD_RAW_DIR before calling this script"
  exit 1
fi

if [[ -z "$SWITCHBOARD_EXTRACT_DIR" ]]; then
  echo "Please set SWITCHBOARD_EXTRACT_DIR before calling this script"
  exit 2
fi

if [[ -z "$SWITCHBOARD_SHARD_DIR" ]]; then
  echo "Please set SWITCHBOARD_SHARD_DIR before calling this script"
  exit 3
fi

if [[ -z "$SWITCHBOARD_META_DIR" ]]; then
  echo "Please set SWITCHBOARD_META_DIR before calling this script"
  exit 4
fi

# if NUM_CPU is not set, use all of them by default
if [ -z ${NUM_CPU+x} ]; then
  NUM_CPU=$(nproc) # or fewer if you want to use the PC for other stuff...
fi

# make sure all potential directories exist
mkdir -p "$SWITCHBOARD_RAW_DIR" "$SWITCHBOARD_EXTRACT_DIR" "$SWITCHBOARD_SHARD_DIR" "$SWITCHBOARD_META_DIR"

echo "SWITCHBOARD_RAW_DIR=$LIBRISPEECH_RAW_DIR"
echo "SWITCHBOARD_EXTRACT_DIR=$SWITCHBOARD_EXTRACT_DIR"
echo "SWITCHBOARD_SHARD_DIR=$SWITCHBOARD_SHARD_DIR"
echo "SWITCHBOARD_META_DIR=$SWITCHBOARD_META_DIR"
echo "NUM_CPU=$NUM_CPU"

# setup
poetry update
poetry install

# move to folder of this script so path to all scripts are known
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR" || exit

# execute ALL steps in order :)

# extract
poetry run ./untar_switchboard.sh

# transform to utterances
poetry run ./transform_into_utterances.sh

# write default shards
poetry run ./default_setup_shards.sh
poetry run ./default_write_shards.sh
poetry run ./default_generate_meta_files.sh
