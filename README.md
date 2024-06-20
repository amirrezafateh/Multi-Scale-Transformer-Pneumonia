# Pneumonia Chest X-ray Classification with MSR Model

This repository contains code to train a model for classifying chest X-ray images into normal or pneumonia using a ResNet backbone.

## Requirements

- Python 3.10
- PyTorch
- torchvision

# Dataset

The dataset used for this project is the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray).

## Download and Prepare Dataset

1. Download the dataset
wget -O downloaded_file.zip "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/f12eaf6d-6023-432f-acc9-80c9d7393433/file_downloaded"

2. Unzip the dataset
unzip downloaded_file.zip -d /content/data

3. Verify the dataset structure
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
python train.py --bsz 64 --lr 0.0005 --niter 50 --layers 101 --data_dir /content/data/chest_xray --logpath log

## Arguments

--bsz: Batch size for training (default: 32)
--lr: Learning rate for optimizer (default: 0.001)
--niter: Number of training iterations (epochs) (default: 30)
--layers: Number of layers in ResNet backbone (e.g., 50 or 101) (default: 50)
--data_dir: Directory where the dataset is located (default: ./data)
--logpath: Directory to save the best model checkpoint (default: log)

