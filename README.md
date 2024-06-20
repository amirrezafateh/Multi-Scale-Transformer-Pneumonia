# Pneumonia Chest X-ray Classification with MSR Model

This repository contains code to train a model for classifying chest X-ray images into normal or pneumonia using a ResNet backbone.

## Requirements

- Python 3.10
- PyTorch
- torchvision

# Dataset

The dataset used for this project is the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray).

## Prepare Dataset

Verify the dataset structure
Make sure your dataset is organized as follows:

    /data/chest_xray
    ├── train
    │   ├── NORMAL
    │   ├── PNEUMONIA
    └── test
        ├── NORMAL
        ├── PNEUMONIA

If your dataset structure is different, adjust the --data_dir argument accordingly.

# Training
To train the model, use the following command:

> ```bash
> python train.py --bsz 64
>                 --lr 1e-4
>                 --niter 50
>                 -layers {50,101}
>                 --data_dir /content/data/chest_xray
>                 --logpath "your_experiment_name"
> ```


## Arguments

- `--bsz`: Batch size for training. Default: `32`
- `--lr`: Learning rate for the optimizer. Default: `0.001`
- `--niter`: Number of training iterations (epochs). Default: `30`
- `--layers`: Number of layers in the ResNet backbone (e.g., 50 or 101). Default: `50`
- `--data_dir`: Directory where the dataset is located. Default: `./data`
- `--logpath`: Directory to save the best model checkpoint. Default: `log`

