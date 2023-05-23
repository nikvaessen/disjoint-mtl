# Switchboard

These instructions are intended to use the 2000 HUB5 English Evaluation Speech
as an evaluation set for ASR.

## environment

```bash
HUB5_ROOT_DIR=${PWD}/data/hub5/
HUB5_RAW_DIR=${HUB5_ROOT_DIR}/raw
HUB5_SHARD_DIR=${HUB5_ROOT_DIR}/shards
HUB5_EXTRACT_DIR=${PWD}/extract
```

## Data acquisition

Place `hub5e_00_LDC2002S09.tgz` and `LDC2002T43.tgz` in `$HUB5_RAW_DIR`.
These files can be obtained through https://catalog.ldc.upenn.edu/LDC2002S09 and https://catalog.ldc.upenn.edu/LDC2002T43.


