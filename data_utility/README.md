# Data utility for speech-related tasks

This repo provides the utility to prepare the following datasets to a format useful 
for training and evaluating deep neural networks. The provided utility encompasses:

1. downloading datasets
2. generating splits such as train/val/dev/test
3. Writing data into shards for efficient data loading 
4. Providing building blocks for `torchdata` datapipes for the ASR and SV task.
5. Evaluating the ASR and SV tasks.

## Datasets

The following datasets are provided:

1. LibriSpeech (see [readme.md](sets/librispeech/readme.md))
2. VoxCeleb (see [readme.md](sets/voxceleb/readme.md))

## Data structure

For efficient i/o on network-attached storage, dataset splits are presented as a set 
of shards. A shard is a tar file containing `n` samples, in the following manner:

```
├── 0
│   └── dataset_id
│       └── speaker_id
│           └── recording_id
│               ├── utterance_id.json
│               └── utterance_id.wav

├── ...
│   └── dataset_id
│       └── speaker_id
│           └── recording_id
│               ├── utterance_id.json
│               └── utterance_id.wav
└── n
    └── dataset_id
        └── speaker_id
            └── recording_id
                ├── utterance_id.json
                └── utterance_id.wav
```

As you can see, a shard contains `n` folders, where each folder contains a single `json` and `wav` file with the same stem name.

The JSON file contains ground truth information (gender, transcript, speaker_id), and meta-information (num_frames, sample_rate).

The JSON of a sample from respectively vox2 and LibriSpeech:

```json
{"num_frames": 345088, "sample_rate": 16000, "gender": "f", "transcription": null, "speaker_id": "vc2/id06229", "sample_id": "vc2/id06229/iRWUcEgx6nM/00124"}
```

```json
{"num_frames": 101200, "sample_rate": 16000, "gender": "m", "transcription": "after the bell had been rolled into the swamp there was of course no more chance of ringing it in such wise as to break it", "speaker_id": "ls/2952", "sample_id": "ls/2952/407/0019"}
```

To write shards, a dataset split is defined in the following CSV format:

```
   key                   path               num_frames  sample_rate speaker_id recording_id gender  transcription
0  ls/3259/158083/0000  /path/to/audio/file 131199      16000       ls/3259    ls/158083    f       administration terrorism ...
1  ls/3259/158083/0001  /path/to/audio/file 247200      16000       ls/3259    ls/158083    f       the agitation would ...
2  ls/3259/158083/0002  /path/to/audio/file 246640      16000       ls/3259    ls/158083    f       forty one women ...
```

Note that each sample has a key, in the format `${dataset_id}/${speaker_id}/${recording_id}/${utterance_id}`, which
matches the path within the tar file of a shard.

These CSV files can be generated for voxceleb with `voxceleb/generate_csv.py` and for librispeech with `librispeech/generate.csv.py`.

### Utility scripts

We provide the following utility scripts for wrangling the datasets into the shards and 
providing the required meta-files for training. 
These are automatically loaded as executables in `$PATH` when `poetry shell` is used.
They can also be run with `poetry run $scriptname <arguments>`.

* `collect_statistics`
* `convert_to_wav`
* `generate_character_distribution`
* `generate_character_vocabulary`
* `generate_speaker_mapping`
* `generate_speaker_trials`
* `split_csv`
* `verify_checksum`
* `write_tar_shards`

For more information, see the `$scriptname --help` or read the file at `data_utility/scripts/${scriptname}.py`.
