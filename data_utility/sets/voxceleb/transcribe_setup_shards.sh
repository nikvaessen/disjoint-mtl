#! /usr/bin/env bash
set -e

# check whether paths have been set
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
mkdir -p "$VOXCELEB_EXTRACT_DIR" "$VOXCELEB_SHARD_DIR" "$VOXCELEB_META_DIR"

# folders for each split
vox2_dev=$VOXCELEB_SHARD_DIR/transcribe/vox2_dev
vox2_test=$VOXCELEB_SHARD_DIR/transcribe/vox2_test

vox1_dev=$VOXCELEB_SHARD_DIR/transcribe/vox1_dev
vox1_test=$VOXCELEB_SHARD_DIR/transcribe/vox1_test

# make folders
mkdir -p "$vox2_dev" "$vox2_test" "$vox1_dev" "$vox1_test"

# make the CSV file for vox2-dev
vc_generate_csv \
"$VOXCELEB_EXTRACT_DIR"/voxceleb2/train/wav \
--csv "$vox2_dev"/_meta.vox2_dev.csv \
--meta "$VOXCELEB_META_DIR"/vox_meta.csv \
--ext wav

# make the CSV file for vox2-test
vc_generate_csv \
"$VOXCELEB_EXTRACT_DIR"/voxceleb2/test/wav \
--csv "$vox2_test"/_meta.vox2_test.csv \
--meta "$VOXCELEB_META_DIR"/vox_meta.csv \
--ext wav

# make the CSV file for vox1-dev
vc_generate_csv \
"$VOXCELEB_EXTRACT_DIR"/voxceleb1/train/wav \
--csv "$vox1_dev"/_meta.vox1_dev.csv \
--meta "$VOXCELEB_META_DIR"/vox_meta.csv \
--ext wav

# make the CSV file for vox1-test
vc_generate_csv \
"$VOXCELEB_EXTRACT_DIR"/voxceleb1/test/wav \
--csv "$vox1_test"/_meta.vox1_test.csv \
--meta "$VOXCELEB_META_DIR"/vox_meta.csv \
--ext wav
