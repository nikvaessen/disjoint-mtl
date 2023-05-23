STATS_DIR=./stats
mkdir -p $STATS_DIR

# dirs exist
if [[ -z "$VOXCELEB_SHARD_DIR" ]]; then
  echo "Please set VOXCELEB_SHARD_DIR before calling this script"
  exit 3
fi
if [[ -z "$LIBRISPEECH_SHARD_DIR" ]]; then
  echo "Please set LIBRISPEECH_SHARD_DIR before calling this script"
  exit 3
fi

# if NUM_CPU is not set, use all of them by default
if [ -z ${NUM_CPU+x} ]; then
  NUM_CPU=$(nproc) # or fewer if you want to use the PC for other stuff...
fi
echo "Using $NUM_CPU cpus"

# voxceleb
collect_statistics "$VOXCELEB_SHARD_DIR"/vox2_train  --workers "$NUM_CPU" --partial true > $STATS_DIR/vox2_train.txt
collect_statistics "$VOXCELEB_SHARD_DIR"/vox2_val    --workers "$NUM_CPU" --partial true > $STATS_DIR/vox2_val.txt
collect_statistics "$VOXCELEB_SHARD_DIR"/vox2_dev    --workers "$NUM_CPU" --partial true > $STATS_DIR/vox2_dev.txt
collect_statistics "$VOXCELEB_SHARD_DIR"/vox1_test_o --workers "$NUM_CPU" --partial true > $STATS_DIR/vox1_test_o.txt
collect_statistics "$VOXCELEB_SHARD_DIR"/vox1_test_e --workers "$NUM_CPU" --partial true > $STATS_DIR/vox1_test_e.txt
collect_statistics "$VOXCELEB_SHARD_DIR"/vox1_test_h --workers "$NUM_CPU" --partial true > $STATS_DIR/vox1_test_h.txt

# librispeech
collect_statistics "$LIBRISPEECH_SHARD_DIR"/train-*    --workers "$NUM_CPU" --partial true > $STATS_DIR/ls_960h_train.txt
collect_statistics "$LIBRISPEECH_SHARD_DIR"/val-*      --workers "$NUM_CPU" --partial true > $STATS_DIR/ls_960h_val.txt
collect_statistics "$LIBRISPEECH_SHARD_DIR"/dev-clean  --workers "$NUM_CPU" --partial true > $STATS_DIR/ls_dev_clean.txt
collect_statistics "$LIBRISPEECH_SHARD_DIR"/dev-other  --workers "$NUM_CPU" --partial true > $STATS_DIR/ls_dev_other.txt
collect_statistics "$LIBRISPEECH_SHARD_DIR"/test-clean --workers "$NUM_CPU" --partial true > $STATS_DIR/ls_test_clean.txt
collect_statistics "$LIBRISPEECH_SHARD_DIR"/test-other --workers "$NUM_CPU" --partial true > $STATS_DIR/ls_test_other.txt
