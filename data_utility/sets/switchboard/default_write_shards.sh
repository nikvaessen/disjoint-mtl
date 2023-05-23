#! /usr/bin/env bash
set -e

if [[ -z "$SWITCHBOARD_SHARD_DIR" ]]; then
  echo "Please set SWITCHBOARD_SHARD_DIR before calling this script"
  exit 3
fi


# set the number of parallel processes writing shards to disk
workers=2

# write the train shards
write_tar_shards \
--csv "$SWITCHBOARD_SHARD_DIR"/test/_meta.test.csv \
--out "$SWITCHBOARD_SHARD_DIR"/test \
--strategy random \
--prefix sw \
--workers="$workers"
