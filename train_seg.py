import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from model.UNet import UNet  # Ensure UNet class is defined in model/unet.py
from data.data_loader_segmentation import get_data_loader
from torchvision.utils import save_image

def dice_coefficient(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    if union.item() == 0:
        return 1.0
    
    return 2.0 * intersection / union


def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0  # Track Dice coefficient

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Compute Dice coefficient
        dice = dice_coefficient(outputs, targets)
        running_dice += dice.item()

        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {running_loss/(batch_idx+1):.3f}, Dice: {running_dice/(batch_idx+1):.3f}')

    train_loss = running_loss / len(train_loader)
    train_dice = running_dice / len(train_loader)  # Average Dice coefficient
    return train_loss, train_dice


def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0  # Track Dice coefficient
    val_outputs = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_outputs.append(outputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # Compute Dice coefficient
            dice = dice_coefficient(outputs, targets)
            running_dice += dice.item()
    # Save validation output masks
    save_dir = 'validation_output_masks'
    os.makedirs(save_dir, exist_ok=True)
    for i, output in enumerate(val_outputs):
        save_image(output, os.path.join(save_dir, f'epoch_{epoch}_batch_{i}_output.png'))

    val_loss = running_loss / len(val_loader)
    val_dice = running_dice / len(val_loader)  # Average Dice coefficient
    return val_loss, val_dice


def main(args):
    print(":=========== Pneumonia Chest X-ray Segmentation ===========")
    print(f"|             datapath: {args.data_dir}")
    print(f"|              logpath: {args.logpath}")
    print(f"|                  bsz: {args.bsz}")
    print(f"|                   lr: {args.lr}")
    print(f"|                niter: {args.niter}")
    print(":==========================================================")

    # Get the data loaders for the Pneumonia dataset
    train_loader, val_loader = get_data_loader(args.data_dir, args.bsz)

    # Instantiate the UNet model
    model = UNet(in_channels=1, out_channels=1, feature_dims=64)  # UNet for binary segmentation

    # Print the number of parameters
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"\nTotal # param.: {total_params}")
    print(f"Learnable # param.: {trainable_params}\n")

    # Move the model to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss with logits
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')

    # Train the model
    for epoch in range(args.niter):  # Train for the specified number of epochs
        print(f'Epoch {epoch+1}/{args.niter}')
        start_time = time.time()
        
        train_loss, train_dice = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device, epoch)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Dice: {val_dice:.4f}')
        print(f'Epoch {epoch+1} completed in {epoch_time:.2f} seconds\n')
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.logpath, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.logpath, 'best_model.pth'))
            print(f'Best model saved with loss: {best_val_loss:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet model on Pneumonia chest X-ray segmentation dataset")
    parser.add_argument('--bsz', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--niter', type=int, default=30, help='Number of training iterations (epochs)')
    parser.add_argument('--data_dir', type=str, default='segmentation', help='Directory where dataset is located')
    parser.add_argument('--logpath', type=str, default='log', help='Directory to save the best model checkpoint')
    
    args = parser.parse_args()
    main(args)
