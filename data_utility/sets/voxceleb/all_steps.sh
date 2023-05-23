#! /usr/bin/env bash
set -e

# check all environment variables
if [[ -z "$VOXCELEB_RAW_DIR" ]]; then
  echo "Please set VOXCELEB_RAW_DIR before calling this script"
  exit 1
fi

if [[ -z "$VOXCELEB_EXTRACT_DIR" ]]; then
  echo "Please set VOXCELEB_EXTRACT_DIR before calling this script"
  exit 2
fi

if [[ -z "$VOXCELEB_SHARD_DIR" ]]; then
  echo "Please set VOXCELEB_SHARD_DIR before calling this script"
  exit 3
fi

if [[ -z "$VOXCELEB_META_DIR" ]]; then
  echo "Please set VOXCELEB_META_DIR before calling this script"
  exit 4
fi

# make sure all potential directories exist
mkdir -p "$VOXCELEB_RAW_DIR" "$VOXCELEB_EXTRACT_DIR" "$VOXCELEB_SHARD_DIR" "$VOXCELEB_META_DIR"

echo "VOXCELEB_RAW_DIR=$VOXCELEB_RAW_DIR"
echo "VOXCELEB_EXTRACT_DIR=$VOXCELEB_EXTRACT_DIR"
echo "VOXCELEB_SHARD_DIR=$VOXCELEB_SHARD_DIR"
echo "VOXCELEB_META_DIR=$VOXCELEB_META_DIR"

# move to folder of this script so path to all scripts are known
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR" || exit

# execute ALL steps in order :)

# download meta (data archives need to be done manually)
poetry run ./download_voxceleb_meta.sh
poetry run ./download_voxceleb_transcripts.sh

# extract
poetry run ./unzip_voxceleb_archives.sh

# write default shards
poetry run ./default_setup_shards.sh
poetry run ./default_write_shards.sh
poetry run ./default_generate_meta_files.sh

# write shards for transcription
#poetry run ./transcribe_setup_shards.sh
#poetry run ./transcribe_write_shards.sh