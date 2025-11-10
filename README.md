# Landslide Detection

## Introduction

This project focuses on the detection of landslides using satellite imagery.
The main goal is to create a model that can predict the probability of a landslide occurring in a given area.
The model is trained using a dataset of satellite images chunked into patches, each with an associated mask.

## Requirements

This project has been tested on Linux, WSL, and macOS. There is **no guarantee** that it will work on vanilla Windows.
It uses **uv** to manage dependencies, as it is generally more stable and faster than pip or conda.

To install dependencies, run:

```bash
uv sync --all-extras
```

Next, fetch the data and:

```bash
mkdir -p data/raw                         # create data directory
unzip -qq -o data/raw/land-anomalies.zip -d data/raw   # unzip the data archive
rm data/raw/land-anomalies.zip            # [Optional] remove the archive to save space
```

## Code and How to Run an Experiment

The main code is in `scripts/recipe.py`. Other relevant components:

* **Model registry**: `landslide/model/__init__.py` — defines pretrained model configs and paths to weights
* **Dataloader**: `landslide/data.py`
* **Base run config**: `config/base.yml`

Example run:

```bash
uv run scripts/recipe.py --config 'config/base.yml' --name 'run_000' --model 'L4S/unet' --dataset 'data/processed/csv_split/A19_5cm.yml' --criterion 'bce' --device 'mps:0' --image_sz 128 --mask_sz 128
```

It is important to customize the dataset path and device.

To track an experiment it is need to set WANDB_API_KEY as env variable and to add the flag `--tracker wandb` to execution command

## Reports

* [Criterion Evaluation Report](https://wandb.ai/gianluca-calo11/landslide/reports/Criterion-evaluation--VmlldzoxNDk1NzE4Nw)
* [Model Evaluation Report](https://wandb.ai/gianluca-calo11/landslide/reports/Model-comparison--VmlldzoxNTAwOTcxNQ)
* [Project Report](https://github.com/lum1t4/landslide/blob/main/docs/Final%20Report.pdf)
