python generate_sbatch.py \
--dir "$VOXCELEB_SHARD_DIR"/transcribe \
--out whisper_base_vox.sbatch \
--model base \
--cache "$VOXCELEB_META_DIR"/transcripts/base.transcript.json