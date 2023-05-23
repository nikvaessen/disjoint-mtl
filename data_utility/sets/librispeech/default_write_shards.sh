#! /usr/bin/env bash
set -e

# check whether paths have been set
if [[ -z "$LIBRISPEECH_SHARD_DIR" ]]; then
  echo "Please set LIBRISPEECH_SHARD_DIR before calling this script"
  exit 3
fi

# make sure all potential directories exist
mkdir -p "$LIBRISPEECH_SHARD_DIR"

# folders for each split
train_clean_100=$LIBRISPEECH_SHARD_DIR/train-clean-100
train_clean_360=$LIBRISPEECH_SHARD_DIR/train-clean-360
train_other_500=$LIBRISPEECH_SHARD_DIR/train-other-500

val_clean_100=$LIBRISPEECH_SHARD_DIR/val-clean-100
val_clean_360=$LIBRISPEECH_SHARD_DIR/val-clean-360
val_other_500=$LIBRISPEECH_SHARD_DIR/val-other-500

dev_clean=$LIBRISPEECH_SHARD_DIR/dev-clean
dev_other=$LIBRISPEECH_SHARD_DIR/dev-other

test_clean=$LIBRISPEECH_SHARD_DIR/test-clean
test_other=$LIBRISPEECH_SHARD_DIR/test-other

# set the number of parallel processes writing shards to disk
workers=2

# write the train shards
write_tar_shards \
--csv "$train_clean_100"/_meta.train_clean_100.csv \
--out "$train_clean_100" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix train_clean_100 \
--workers="$workers"

write_tar_shards \
--csv "$train_clean_360"/_meta.train_clean_360.csv \
--out "$train_clean_360" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix train_clean_360 \
--workers="$workers"

write_tar_shards \
--csv "$train_other_500"/_meta.train_other_500.csv \
--out "$train_other_500" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix train_other_500 \
--workers="$workers"

# write val shards
write_tar_shards \
--csv "$val_clean_100"/_meta.val_clean_100.csv \
--out "$val_clean_100" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix val_clean_100 \
--workers="$workers"

write_tar_shards \
--csv "$val_clean_360"/_meta.val_clean_360.csv \
--out "$val_clean_360" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix val_clean_360 \
--workers="$workers"

write_tar_shards \
--csv "$val_other_500"/_meta.val_other_500.csv \
--out "$val_other_500" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix val_other_500 \
--workers="$workers"

# write the dev shards
write_tar_shards \
--csv "$dev_clean"/_meta.dev_clean.csv \
--out "$dev_clean" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix dev_clean \
--workers="$workers"

write_tar_shards \
--csv "$dev_other"/_meta.dev_other.csv \
--out "$dev_other" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix dev_other \
--workers="$workers"

# write the test shards
write_tar_shards \
--csv "$test_clean"/_meta.test_clean.csv \
--out "$test_clean" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix test_clean \
--workers="$workers"

write_tar_shards \
--csv "$test_other"/_meta.test_other.csv \
--out "$test_other" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix test_other \
--workers="$workers"
