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
vox2_train=$VOXCELEB_SHARD_DIR/vox2_train
vox2_val=$VOXCELEB_SHARD_DIR/vox2_val
vox2_dev=$VOXCELEB_SHARD_DIR/vox2_dev

vox1_test_o=$VOXCELEB_SHARD_DIR/vox1_test_o
vox1_test_e=$VOXCELEB_SHARD_DIR/vox1_test_e
vox1_test_h=$VOXCELEB_SHARD_DIR/vox1_test_h

# make folders
mkdir -p "$vox2_train" "$vox2_val" "$vox2_dev"
mkdir -p "$vox1_test_o" "$vox1_test_e" "$vox1_test_h"

# make the CSV file for vox2-dev
vc_generate_csv \
"$VOXCELEB_EXTRACT_DIR"/voxceleb2/train/wav \
--csv "$vox2_train"/_meta.vox2_train.csv \
--meta "$VOXCELEB_META_DIR"/vox_meta.csv \
--transcript "$VOXCELEB_META_DIR"/transcripts/vox2.dev.base.transcripts.json \
--enforce_en_vocab true \
--ext wav

# make the CSV file for vox1-test-original
vc_generate_csv \
"$VOXCELEB_EXTRACT_DIR"/voxceleb1/test/wav \
--csv "$vox1_test_o"/_meta.vox1_test_o.csv \
--meta "$VOXCELEB_META_DIR"/vox_meta.csv \
--trials "$VOXCELEB_META_DIR"/veri_test2.txt \
--transcript "$VOXCELEB_META_DIR"/transcripts/vox1.base.transcript.json \
--enforce_en_vocab true \
--ext wav

# make the CSV file for vox1-test-everyone
vc_generate_csv \
"$VOXCELEB_EXTRACT_DIR"/voxceleb1/train/wav "$VOXCELEB_EXTRACT_DIR"/voxceleb1/test/wav  \
--csv "$vox1_test_e"/_meta.vox1_test_e.csv \
--meta "$VOXCELEB_META_DIR"/vox_meta.csv \
--trials "$VOXCELEB_META_DIR"/list_test_all2.txt \
--transcript "$VOXCELEB_META_DIR"/transcripts/vox1.base.transcript.json \
--enforce_en_vocab true \
--ext wav

# make the CSV file for vox1-test-hard
vc_generate_csv \
"$VOXCELEB_EXTRACT_DIR"/voxceleb1/train/wav "$VOXCELEB_EXTRACT_DIR"/voxceleb1/test/wav  \
--csv "$vox1_test_h"/_meta.vox1_test_h.csv \
--meta "$VOXCELEB_META_DIR"/vox_meta.csv \
--trials "$VOXCELEB_META_DIR"/list_test_hard2.txt \
--transcript "$VOXCELEB_META_DIR"/transcripts/vox1.base.transcript.json \
--enforce_en_vocab true \
--ext wav

## split the vox2 train split into train/dev
DEV_RATIO=0.03
split_csv "$vox2_train"/_meta.vox2_train.csv  \
--delete-in \
--strategy by_speakers_equal --ratio $DEV_RATIO \
--remain-out "$vox2_train"/_meta.vox2_train.csv \
--split-out "$vox2_dev"/_meta.vox2_dev.csv

# further split the vox2 train split into train/val
VAL_RATIO=0.03
split_csv "$vox2_train"/_meta.vox2_train.csv  \
--delete-in \
--strategy by_recordings --ratio $VAL_RATIO \
--remain-out "$vox2_train"/_meta.vox2_train.csv \
--split-out "$vox2_val"/_meta.vox2_val.csv
