#! /usr/bin/env bash
set -e

# check all environment variables
if [[ -z "$LIBRISPEECH_RAW_DIR" ]]; then
  echo "Please set LIBRISPEECH_RAW_DIR before calling this script"
  exit 1
fi

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

# if NUM_CPU is not set, use all of them by default
if [ -z ${NUM_CPU+x} ]; then
  NUM_CPU=$(nproc) # or fewer if you want to use the PC for other stuff...
fi

# make sure all potential directories exist
mkdir -p "$LIBRISPEECH_RAW_DIR" "$LIBRISPEECH_EXTRACT_DIR" "$LIBRISPEECH_SHARD_DIR" "$LIBRISPEECH_META_DIR"

echo "LIBRISPEECH_RAW_DIR=$LIBRISPEECH_RAW_DIR"
echo "LIBRISPEECH_EXTRACT_DIR=$LIBRISPEECH_EXTRACT_DIR"
echo "LIBRISPEECH_SHARD_DIR=$LIBRISPEECH_SHARD_DIR"
echo "LIBRISPEECH_META_DIR=$LIBRISPEECH_META_DIR"
echo "NUM_CPU=$NUM_CPU"

# check ffmpeg is installed
if ! [ -x "$(command -v ffmpeg)" ]; then
  echo 'Error: ffmpeg is not installed.' >&2
  exit 5
fi

# move to folder of this script so path to all scripts are known
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR" || exit

# execute ALL steps in order :)

# download
poetry run ./download_librispeech.sh

# extract
poetry run ./untar_librispeech_archives.sh

# convert to wav
poetry run convert_to_wav \
  --dir "$LIBRISPEECH_EXTRACT_DIR" \
  --ext .flac \
  --workers="$NUM_CPU"

# write default shards
poetry run ./default_setup_shards.sh
poetry run ./default_write_shards.sh
poetry run ./default_generate_meta_files.sh
