#! /usr/bin/env bash
set -e

# check whether paths have been set
if [[ -z "$LIBRISPEECH_RAW_DIR" ]]; then
  echo "Please set LIBRISPEECH_RAW_DIR before calling this script"
  exit 1
fi

if [[ -z "$LIBRISPEECH_EXTRACT_DIR" ]]; then
  echo "Please set LIBRISPEECH_EXTRACT_DIR before calling this script"
  exit 2
fi

if [[ -z "$LIBRISPEECH_META_DIR" ]]; then
  echo "Please set LIBRISPEECH_META_DIR before calling this script"
  exit 4
fi

# make sure all potential directories exist
mkdir -p "$LIBRISPEECH_RAW_DIR" "$LIBRISPEECH_EXTRACT_DIR" "$LIBRISPEECH_META_DIR"

# define paths to tar files and output location
UNZIP_DIR=$LIBRISPEECH_EXTRACT_DIR
DATA_DIR=$LIBRISPEECH_RAW_DIR

TRAIN_CLEAN_100="$DATA_DIR"/train-clean-100.tar.gz
TRAIN_CLEAN_360="$DATA_DIR"/train-clean-360.tar.gz
TRAIN_OTHER_500="$DATA_DIR"/train-other-500.tar.gz
DEV_CLEAN="$DATA_DIR"/dev-clean.tar.gz
DEV_OTHER="$DATA_DIR"/dev-other.tar.gz
TEST_CLEAN="$DATA_DIR"/test-clean.tar.gz
TEST_OTHER="$DATA_DIR"/test-other.tar.gz

# make sure unzip dir exists
mkdir -p "$UNZIP_DIR"

# train clean 100
if [ -f "$TRAIN_CLEAN_100" ]; then
    echo "extracting $TRAIN_CLEAN_100"
    # NUM_FILES=$(gzip -cd "$TRAIN_CLEAN_100" | tar -tvv | grep -c ^-)
    NUM_FILES=29966 # 29129
    tar xzfv "$TRAIN_CLEAN_100" -C "$UNZIP_DIR" | tqdm --total "$NUM_FILES" >> /dev/null
else
   echo "file $TRAIN_CLEAN_100 does not exist."
fi

# train clean 360
if [ -f "$TRAIN_CLEAN_360" ]; then
    echo "extracting $TRAIN_CLEAN_360"
    # NUM_FILES=$(gzip -cd "$TRAIN_CLEAN_360" | tar -tvv | grep -c ^-)
    NUM_FILES=109135 # 106116
    tar xzfv "$TRAIN_CLEAN_360" -C "$UNZIP_DIR" | tqdm --total "$NUM_FILES" >> /dev/null
else
   echo "file $TRAIN_CLEAN_360 does not exist."
fi

# train other 500
if [ -f "$TRAIN_OTHER_500" ]; then
    echo "extracting $TRAIN_OTHER_500"
    # NUM_FILES=$(gzip -cd "$TRAIN_OTHER_500" | tar -tvv | grep -c ^-)
    NUM_FILES=155428 # 151477
    tar xzfv "$TRAIN_OTHER_500" -C "$UNZIP_DIR"| tqdm --total "$NUM_FILES" >> /dev/null
else
   echo "file $TRAIN_OTHER_500 does not exist."
fi

# dev clean
if [ -f "$DEV_CLEAN" ]; then
    echo "extracting $DEV_CLEAN"
    # NUM_FILES=$(gzip -cd "$DEV_CLEAN" | tar -tvv | grep -c ^-)
    NUM_FILES=2943 # 2805
    tar xzfv "$DEV_CLEAN" -C "$UNZIP_DIR" | tqdm --total "$NUM_FILES" >> /dev/null
else
   echo "file $DEV_CLEAN does not exist."
fi

# dev other
if [ -f "$DEV_OTHER" ]; then
    echo "extracting $DEV_OTHER"
    # NUM_FILES=$(gzip -cd "$DEV_OTHER" | tar -tvv | grep -c ^-)
    NUM_FILES=3085 # 2960
    tar xzfv "$DEV_OTHER" -C "$UNZIP_DIR" | tqdm --total "$NUM_FILES" >> /dev/null
else
   echo "file $DEV_OTHER does not exist."
fi

# test clean
if [ -f "$TEST_CLEAN" ]; then
    echo "extracting $TEST_CLEAN"
    # NUM_FILES=$(gzip -cd "$TEST_CLEAN" | tar -tvv | grep -c ^-)
    NUM_FILES=2840 # 2712
    tar xzfv "$TEST_CLEAN" -C "$UNZIP_DIR"| tqdm --total "$NUM_FILES" >> /dev/null
else
   echo "file $TEST_CLEAN does not exist."
fi

# test other
if [ -f "$TEST_OTHER" ]; then
    echo "extracting $TEST_OTHER"
    # NUM_FILES=$(gzip -cd "$TEST_OTHER" | tar -tvv | grep -c ^-)
    NUM_FILES=3158 # 3034
    tar xzfv "$TEST_OTHER" -C "$UNZIP_DIR" | tqdm --total "$NUM_FILES" >> /dev/null
else
   echo "file $TEST_OTHER does not exist."
fi

# copy meta information files
cp "$LIBRISPEECH_EXTRACT_DIR"/LibriSpeech/BOOKS.TXT "$LIBRISPEECH_META_DIR"/_archive_books.txt
cp "$LIBRISPEECH_EXTRACT_DIR"/LibriSpeech/CHAPTERS.TXT "$LIBRISPEECH_META_DIR"/_archive_chapters.txt
cp "$LIBRISPEECH_EXTRACT_DIR"/LibriSpeech/SPEAKERS.TXT "$LIBRISPEECH_META_DIR"/_archive_speakers.txt