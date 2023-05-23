# Disjoint MTL

Code accompanying paper "Towards multi-task learning for speech and speaker recognition"

See [paper_experiments.md](paper_experiments.md) for commands to reproduce results.

See [here](https://surfdrive.surf.nl/files/index.php/s/YINHj9yBcxmy3cE) for some model checkpoints.

See [here](https://surfdrive.surf.nl/files/index.php/s/KHBa8P0q4uhybKh) for VoxCeleb 1/2 ASR labels with [Whisper](https://github.com/openai/whisper).

# Quick start guide

Copy `.env.example` to `.env` and fill accordingly.

See [data_utility](data_utility) for instructions for preparing data. See
[sre2008](data_utility%2Fsets%2Fsre2008),
[hub5_2000](data_utility%2Fsets%2Fhub5_2000),
[voxceleb](data_utility%2Fsets%2Fvoxceleb) and
[librispeech](data_utility%2Fsets%2Flibrispeech).

Install dependencies with `poetry update`.

Run experiments with [run_mtl_disjoint.py](run_mtl_disjoint.py),
[run_mtl_joint.py](run_mtl_joint.py),
[run_speaker.py](run_speaker.py) and
[run_speech.py](run_speech.py).

# Cite

You can cite this work as:

```
@INPROCEEDINGS{vaessen2023mtl,
  author={Vaessen, Nik and van Leeuwen, David A.},
  booktitle={Interspeech 2023}, 
  title={Towards multi-task learning for speech and speaker recognition}, 
  year={2023},
}
```


