#! /usr/bin/env bash
set -e

# check whether paths have been set
if [[ -z "$VOXCELEB_SHARD_DIR" ]]; then
  echo "Please set VOXCELEB_SHARD_DIR before calling this script"
  exit 3
fi

# make sure all potential directories exist
mkdir -p "$VOXCELEB_SHARD_DIR"

# folders for each split
vox2_train=$VOXCELEB_SHARD_DIR/vox2_train
vox2_val=$VOXCELEB_SHARD_DIR/vox2_val
vox2_dev=$VOXCELEB_SHARD_DIR/vox2_dev

vox1_test_o=$VOXCELEB_SHARD_DIR/vox1_test_o
vox1_test_e=$VOXCELEB_SHARD_DIR/vox1_test_e
vox1_test_h=$VOXCELEB_SHARD_DIR/vox1_test_h

# set the number of parallel processes writing shards to disk
workers=2

# write test shards
write_tar_shards \
--csv "$vox1_test_o"/_meta.vox1_test_o.csv \
--out "$vox1_test_o" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix vox1_test_o \
--workers="$workers"

write_tar_shards \
--csv "$vox1_test_e"/_meta.vox1_test_e.csv \
--out "$vox1_test_e" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix vox1_test_e \
--workers="$workers"

write_tar_shards \
--csv "$vox1_test_h"/_meta.vox1_test_h.csv \
--out "$vox1_test_h" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix vox1_test_h \
--workers="$workers"

# write the vox2 train/val/dev shards
write_tar_shards \
--csv "$vox2_val"/_meta.vox2_val.csv \
--out "$vox2_val" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix vox2_val \
--workers="$workers"

write_tar_shards \
--csv "$vox2_dev"/_meta.vox2_dev.csv \
--out "$vox2_dev" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix vox2_dev \
--workers="$workers"

write_tar_shards \
--csv "$vox2_train"/_meta.vox2_train.csv \
--out "$vox2_train" \
--strategy random \
--samples_per_shard 5000 \
--prefix vox2_train \
--workers="$workers"
