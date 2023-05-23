#! /usr/bin/env bash
set -e

# check all environment variables
if [[ -z "$HUB5_RAW_DIR" ]]; then
  echo "Please set HUB5_RAW_DIR before calling this script"
  exit 1
fi

if [[ -z "$HUB5_EXTRACT_DIR" ]]; then
  echo "Please set HUB5_EXTRACT_DIR before calling this script"
  exit 2
fi

# make sure all potential directories exist
mkdir -p "$HUB5_RAW_DIR" "$HUB5_EXTRACT_DIR"

TAR_FILE=$HUB5_RAW_DIR/LDC2002T43.tgz
if [ -f "$TAR_FILE" ]; then
    tar xzfv "$TAR_FILE" -C "$HUB5_EXTRACT_DIR"
else
   echo "file $TAR_FILE does not exist."
fi

TAR_FILE=$HUB5_RAW_DIR/hub5e_00_LDC2002S09.tgz
if [ -f "$TAR_FILE" ]; then
    tar xzfv "$TAR_FILE" -C "$HUB5_EXTRACT_DIR"
else
   echo "file $TAR_FILE does not exist."
fi

