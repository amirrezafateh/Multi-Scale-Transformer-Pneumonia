# Pneumonia Chest X-ray Classification with Multi-Scale Transformer
This is the implementation of the paper "Pneumonia Chest X-ray Classification with Multi-Scale Transformer" by...

<p align="middle">
    <img src="data/overview.png">
</p>

## Requirements

- Python 3.8.18
- PyTorch 1.8.1
- cuda 11.5
- tensorboard 2.12.1
- numpy 1.22.3
- sklearn 1.3

# Dataset

The dataset used for this project is the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray).

## Prepare Dataset

Verify the dataset structure
Make sure your dataset is organized as follows:
                            
    ../                      
    ├── common/             
    ├── data/               
    |   ├── dataloader.py
    |   └── chest_xray/
    │       ├── train/
    │       │   ├── NORMAL
    │       |   └── PNEUMONIA
    │       └── test/
    │           ├── NORMAL
    │           └── PNEUMONIA
    ├── model/
    |   ├── MSR.py
    |   ├── backbone_utils.py
    |   └── transformer.py
    ├── README.md           
    ├── train.py            
    └── test.py             
    


If your dataset structure is different, adjust the --data_dir argument accordingly.

# Training
To train the model, use the following command:

> ```bash
> python train.py --bsz 64
>                 --lr 1e-5
>                 --niter 30
>                 -layers {50,101}
>                 --data_dir data/chest_xray
>                 --logpath "your_experiment_name"
> ```

# Testing
To test the model, use the following command:

> ```bash
> python test.py  --bsz 64
>                 -layers {50,101}
>                 --data_dir data/chest_xray
>                 --logpath "your_experiment_name"
> ```


## Arguments

- `--bsz`: Batch size for training. Default: `32`
- `--lr`: Learning rate for the optimizer. Default: `0.00001`
- `--niter`: Number of training iterations (epochs). Default: `30`
- `--layers`: Number of layers in the ResNet backbone (e.g., 50 or 101). Default: `50`
- `--data_dir`: Directory where the dataset is located. Default: `.data/chest_xray`
- `--logpath`: Directory to save the best model checkpoint. Default: `log`

