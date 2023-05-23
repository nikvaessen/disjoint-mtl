python generate_sbatch.py \
--dir "$VOXCELEB_SHARD_DIR"/transcribe \
--out whisper_large-v2_vox.sbatch \
--model large-v2 \
--cache "$VOXCELEB_META_DIR"/transcripts/large-v2.transcript.json