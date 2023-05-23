#! /usr/bin/env bash
set -e

# check if download location is set
if [[ -z "$LIBRISPEECH_RAW_DIR" ]]; then
  echo "Please set LIBRISPEECH_RAW_DIR before calling this script"
  exit 1
fi

# make sure all potential directories exist
mkdir -p "$LIBRISPEECH_RAW_DIR"

# default directory to save files in
DIR=$LIBRISPEECH_RAW_DIR
echo "Downloading LibriSpeech dataset to $DIR"

## download files
echo "--- Downloading dev set ---"
echo "--- clean"
curl -C - https://www.openslr.org/resources/12/dev-clean.tar.gz --output "$DIR"/dev-clean.tar.gz
echo "--- other"
curl -C - https://www.openslr.org/resources/12/dev-other.tar.gz --output "$DIR"/dev-other.tar.gz

echo "--- Downloading test set ---"
echo "--- clean"
curl -C - https://www.openslr.org/resources/12/test-clean.tar.gz --output "$DIR"/test-clean.tar.gz
echo "--- other"
curl -C - https://www.openslr.org/resources/12/test-other.tar.gz --output "$DIR"/test-other.tar.gz

echo "--- Downloading train set ---"
echo "--- 100h"
curl -C - https://www.openslr.org/resources/12/train-clean-100.tar.gz --output "$DIR"/train-clean-100.tar.gz
echo "--- 360h"
curl -C - https://www.openslr.org/resources/12/train-clean-360.tar.gz --output "$DIR"/train-clean-360.tar.gz
echo "--- 500h"
curl -C - https://www.openslr.org/resources/12/train-other-500.tar.gz --output "$DIR"/train-other-500.tar.gz

# verify checksums
echo "verifying checksums"
verify_checksum "$DIR"/dev-clean.tar.gz 42e2234ba48799c1f50f24a7926300a1 --algo md5
verify_checksum "$DIR"/dev-other.tar.gz c8d0bcc9cca99d4f8b62fcc847357931 --algo md5

verify_checksum "$DIR"/test-clean.tar.gz 32fa31d27d2e1cad72775fee3f4849a9 --algo md5
verify_checksum "$DIR"/test-other.tar.gz fb5a50374b501bb3bac4815ee91d3135 --algo md5

verify_checksum "$DIR"/train-clean-100.tar.gz 2a93770f6d5c6c964bc36631d331a522 --algo md5
verify_checksum "$DIR"/train-clean-360.tar.gz c0e676e450a7ff2f54aeade5171606fa --algo md5
verify_checksum "$DIR"/train-other-500.tar.gz d1a0fd59409feb2c614ce4d30c387708 --algo md5
