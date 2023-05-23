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
vox2_dev=$VOXCELEB_SHARD_DIR/transcribe/vox2_dev
vox2_test=$VOXCELEB_SHARD_DIR/transcribe/vox2_test

vox1_dev=$VOXCELEB_SHARD_DIR/transcribe/vox1_dev
vox1_test=$VOXCELEB_SHARD_DIR/transcribe/vox1_test

# set the number of parallel processes writing shards to disk
workers=2

# write the vox2 train shards
write_tar_shards \
--csv "$vox2_dev"/_meta.vox2_dev.csv \
--out "$vox2_dev" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix vox2_dev \
--compress true \
--workers="$workers"

# write the vox2 test shards
write_tar_shards \
--csv "$vox2_test"/_meta.vox2_test.csv \
--out "$vox2_test" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix vox2_test \
--compress true \
--workers="$workers"

# write the vox1 dev shards
write_tar_shards \
--csv "$vox1_dev"/_meta.vox1_dev.csv \
--out "$vox1_dev" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix vox1_dev \
--compress true \
--workers="$workers"

# write the vox1 test shards
write_tar_shards \
--csv "$vox1_test"/_meta.vox1_test.csv \
--out "$vox1_test" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix vox1_test \
--compress true \
--workers="$workers"
