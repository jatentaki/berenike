# Tumulus detection
This repository contains the code accompanying the paper "A multifaceted spatial analysis of tomb distribution in Blemmyan Berenike (Eastern Desert of Egypt)" by
Mariusz Gwiazda, Anna Fijałkowska, Oskar Graszka, Tomasz Herbich and Michał Tyszkiewicz.

## Installation instructions
Start by cloning the repository. The model checkpoint is tracked via [git-lfs](https://git-lfs.com/) and may require installation if you don't have it already.

This code is formatted as a Python package.
I recommend to use [uv](https://docs.astral.sh/uv/) for its speed but regular `conda` + `pip` installation will also work.

Install `uv`:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a virtual environment in the current directory and activate it
```
uv venv .venv
source .venv/bin/activate
```

Install this package in this venv
```
uv pip install .
```

## Getting the data
Download the data from [this google drive link](https://drive.google.com/drive/folders/1luyfJm7-_6esYJlUozbVVCgfeo8-LBDZ?usp=sharing) and put it in this repository as `./data/...`.
The contents are as follows:
```
data/
|-- Egypt_ortho_8bit.tif # the original satellite image
|-- annotations.geojson # hand-annotated tumuli, exhaustive only within the survey area
|-- train-chunks # a single chunk of the satellite image preprocessed to include binary labels and image validity mask
|   `-- train.npz
|-- val-chunks # preprocessed validation area, same format as `train-chunks` but in multiple pieces
|   |-- 0.npz
|   |-- 1.npz
|   |-- <omitted... >
|   |-- 25.npz
|   |-- 26.npz
`-- validation_area.geojson # a polygon indicating the area used for validation (and not for training)
```


## Running
> :warning: Both training and inference require having [downloaded the data](#getting-the-data)

To train, execute `python train.py` with a GPU with at least 32GB of VRAM and half precision support (e.g. V100).
Training with a smaller batch size (see inside [train.py](train.py)) is also likely to work but wasn't tested.

See the [inference](infer.ipynb) notebook for performing inference and evaluating metrics.
This will run inference with our provided model checkpoint by default (can be swapped to your own trained checkpoints).
