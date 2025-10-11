# Testing of KAN nnue networks

This document explains how to use the scripts to test the KAN nnue networks and allows calculation of ELO points for
them.

## Programs included

| program             | purpose                             |
|---------------------|-------------------------------------|
| cross_check_eval.py | to compare evaluations to stockfish |

## Models and how to test them

You must know the checkpoint files of the models you want to use.
You should add your nets to `nets` in the `models` package.

## Other files

- `CMakeLists.txt` - for building the loader
- `training_data_loader.cpp` - cpp data loader