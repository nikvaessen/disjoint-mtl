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

# make sure all potential directories exist
mkdir -p "$SWITCHBOARD_RAW_DIR" "$SWITCHBOARD_EXTRACT_DIR"

TAR_FILE=$SWITCHBOARD_RAW_DIR/swb1.tar.gz
SUM_FILE=$SWITCHBOARD_RAW_DIR/md5sum.txt
UNZIP_DIR=$SWITCHBOARD_EXTRACT_DIR

if [ -f "$TAR_FILE" ]; then
    checksum=$(awk '{ print $1 }' < "$SUM_FILE")
    echo "verifying checksum of $TAR_FILE matches $checksum"
    verify_checksum "$TAR_FILE" "$checksum"

    echo "extracting $TAR_FILE to $UNZIP_DIR"
    #NUM_FILES=$(gzip -cd "$TAR_FILE" | tar -tvv | grep -c ^-)
    NUM_FILES=8052
    echo "$NUM_FILES"
    tar xzfv "$TAR_FILE" -C "$UNZIP_DIR" | tqdm --total "$NUM_FILES" >> /dev/null
else
   echo "file $TAR_FILE does not exist."
fi
