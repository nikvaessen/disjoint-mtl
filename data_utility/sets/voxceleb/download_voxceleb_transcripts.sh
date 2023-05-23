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

# download zip file of transcripts
if [ ! -f "$DL_DIR"/transcripts.zip ]; then
  curl https://surfdrive.surf.nl/files/index.php/s/KHBa8P0q4uhybKh/download --output "$DL_DIR"/transcripts.zip
fi

# unzip zipfile
unzip -o "$DL_DIR"/transcripts.zip -d "$VOXCELEB_META_DIR"

# make merged versions of the transcripts for convenience
cat "$VOXCELEB_META_DIR"/transcripts/vox*.*.base.transcript.json | jq -sSc add > "$VOXCELEB_META_DIR"/transcripts/base.transcript.json
cat "$VOXCELEB_META_DIR"/transcripts/vox*.*.large-v2.transcript.json | jq -sSc add > "$VOXCELEB_META_DIR"/transcripts/large-v2.transcript.json
