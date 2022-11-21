#! /usr/bin/env bash

# you need to source this file instead of executing it
eval "$(python run_speaker.py -sc install=bash)"

# you need to source this file instead of executing it
eval "$(python run_speech.py -sc install=bash)"

# you need to source this file instead of executing it
eval "$(python run_mtl_disjoint.py -sc install=bash)"

# you need to source this file instead of executing it
eval "$(python run_mtl_joint.py -sc install=bash)"

# you need to source this file instead of executing it
eval "$(python run_fashion_mnist.py -sc install=bash)"
