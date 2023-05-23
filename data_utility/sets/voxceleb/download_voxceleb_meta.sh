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

## download files
curl -C - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt --output "$DL_DIR"/iden_split.txt
curl -C - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt --output "$DL_DIR"/veri_test.txt
curl -C - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt --output "$DL_DIR"/veri_test2.txt
curl -C - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard.txt --output "$DL_DIR"/list_test_hard.txt
curl -C - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard2.txt --output "$DL_DIR"/list_test_hard2.txt
curl -C - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all.txt --output "$DL_DIR"/list_test_all.txt
curl -C - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all2.txt --output "$DL_DIR"/list_test_all2.txt
curl -C - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox1_meta.csv --output "$DL_DIR"/vox1_meta.csv
curl -C - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox2_meta.csv --output "$DL_DIR"/vox2_meta.csv

# normalize CSV files to have same header and separator
cp "$DL_DIR"/vox1_meta.csv "$VOXCELEB_META_DIR"/vox1_meta.csv
cp "$DL_DIR"/vox2_meta.csv "$VOXCELEB_META_DIR"/vox2_meta.csv

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
poetry run python "$SCRIPT_DIR"/fix_meta_csv_file.py \
  --vox1_meta_path "$VOXCELEB_META_DIR"/vox1_meta.csv \
  --vox2_meta_path "$VOXCELEB_META_DIR"/vox2_meta.csv \
  --vox_merged_meta_path "$VOXCELEB_META_DIR"/vox_meta.csv

# fix trials such that they contain the dataset ID
poetry run python "$SCRIPT_DIR"/fix_trials.py \
  --path "$DL_DIR"/veri_test2.txt \
  --out "$VOXCELEB_META_DIR"/veri_test2.txt \
  --ds vc1
poetry run python "$SCRIPT_DIR"/fix_trials.py \
  --path "$DL_DIR"/list_test_all2.txt \
  --out "$VOXCELEB_META_DIR"/list_test_all2.txt \
  --ds vc1
poetry run python "$SCRIPT_DIR"/fix_trials.py \
  --path "$DL_DIR"/list_test_hard2.txt \
  --out "$VOXCELEB_META_DIR"/list_test_hard2.txt \
  --ds vc1