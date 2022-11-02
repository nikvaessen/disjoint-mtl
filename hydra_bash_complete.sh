#! /usr/bin/env bash

# you need to source this file instead of executing it
eval "$(python run_speaker.py -sc install=bash)"

# you need to source this file instead of executing it
eval "$(python run_speech.py -sc install=bash)"

# you need to source this file instead of executing it
eval "$(python run_mt_speech_speaker.py -sc install=bash)"

# you need to source this file instead of executing it
eval "$(python run_fashion_mnist.py -sc install=bash)"
