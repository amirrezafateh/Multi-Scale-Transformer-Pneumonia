import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from MSR import MSR
from data.data_loader import get_data_loader

def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {running_loss/(batch_idx+1):.3f}, Accuracy: {100.*correct/total:.2f}%')

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def main(args):
    print(":=========== Pneumonia Chest X-ray ===========")
    print(f"|             datapath: {args.data_dir}")
    print(f"|              logpath: {args.logpath}")
    print(f"|                  bsz: {args.bsz}")
    print(f"|                   lr: {args.lr}")
    print(f"|                niter: {args.niter}")
    print(f"|               layers: {args.layers}")
    print(":========================================")

    # Get the data loaders for the Pneumonia dataset
    train_loader, val_loader = get_data_loader(args.data_dir, args.bsz)

    # Instantiate the MSR model
    model = MSR(layers=args.layers, num_classes=2)  # Binary classification for Pneumonia dataset

    # Print the number of parameters
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"\nBackbone # param.: {total_params}")
    print(f"Learnable # param.: {trainable_params}\n")

    # Move the model to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': model.get_backbone_params(), 'lr': args.lr},
        {'params': model.get_fc_params(), 'lr': args.lr * 10}
    ])

    best_val_acc = 0.0

    # Train the model
    for epoch in range(args.niter):  # Train for the specified number of epochs
        print(f'Epoch {epoch+1}/{args.niter}')
        start_time = time.time()
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
        print(f'Epoch {epoch+1} completed in {epoch_time:.2f} seconds\n')
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.logpath, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.logpath, 'best_model.pth'))
            print('Best model saved with accuracy: {:.2f}%'.format(best_val_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MSR model with ResNet backbone on Pneumonia dataset")
    parser.add_argument('--bsz', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--niter', type=int, default=30, help='Number of training iterations (epochs)')
    parser.add_argument('--layers', type=int, default=50, help='Number of layers in ResNet backbone (e.g., 50 or 101)')
    parser.add_argument('--data_dir', type=str, default='.data/chest_xray', help='Directory where dataset is located')
    parser.add_argument('--logpath', type=str, default='log', help='Directory to save the best model checkpoint')
    
    args = parser.parse_args()
    main(args)
