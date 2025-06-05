# Landslide detection

## Introduction
This project is focused on the detection of landslides using satellite images.
The main goal is to create a model that can predict the probability of a landslide in a given area
The model will be trained using a dataset of satellite images chuncked into patches each one with an assossiated mask.



## Requirements
- torch
- tqdm
- numpy
- pyyaml
- torchvision


## Training
To train the model, run the following command:

**Example:**
```bash
python scripts/model_train.py --model "unet" --image_sz 128 --criterion "binary_cross_entropy" --dataset "data/processed/L4S" --device "cuda:0" --epochs 3 --lr 1e-5 
```

Adjust the parameters as needed and in particular device according to available accelerators (currently tested only on cuda and mps)
