#! /usr/bin/env bash
set -e

# check if download location is set
if [[ -z "$VOXCELEB_META_DIR" ]]; then
  echo "Please set VOXCELEB_META_DIR before calling this script"
  exit 1
fi

# default directory to save files in
DL_DIR=$VOXCELEB_META_DIR/_download

# make sure all potential directories exist
mkdir -p "$DL_DIR"

# download data
curl -C - https://surfdrive.surf.nl/files/index.php/s/ --output "$DL_DIR"/vox1_dev_wav.zip
curl -C - https://surfdrive.surf.nl/files/index.php/s/ --output "$DL_DIR"/vox2_dev_wav.zip
curl -C - https://surfdrive.surf.nl/files/index.php/s/ --output "$DL_DIR"/vox1_test_wav.zip
curl -C - https://surfdrive.surf.nl/files/index.php/s/ --output "$DL_DIR"/vox2_test_wav.zip
