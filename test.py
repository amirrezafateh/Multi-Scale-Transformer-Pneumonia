import argparse
import os
import torch
from model.MSR import MSR
from data.data_loader import get_test_loader

def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # Adjust if necessary
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

def main(args):
    print(":=========== Pneumonia Chest X-ray Test ===========")
    print(f"|             datapath: {args.data_dir}")
    print(f"|              logpath: {args.logpath}")
    print(f"|                  bsz: {args.bsz}")
    print(f"|               layers: {args.layers}")
    print(":========================================")

    # Load the test data
    test_loader = get_test_loader(args.data_dir, args.bsz)

    # Instantiate the MSR model
    model = MSR(layers=args.layers, num_classes=2)  # Binary classification for Pneumonia dataset

    # Move the model to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the model checkpoint
    checkpoint_path = os.path.join(args.logpath, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f'Model loaded from {checkpoint_path}')
    else:
        print(f'No checkpoint found at {checkpoint_path}')
        return

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Test the model
    test_loss, test_acc = test(model, test_loader, criterion, device)

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MSR model with ResNet backbone on Pneumonia dataset")
    parser.add_argument('--bsz', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--layers', type=int, default=50, help='Number of layers in ResNet backbone (e.g., 50 or 101)')
    parser.add_argument('--data_dir', type=str, default='./data/chest_xray', help='Directory where dataset is located')
    parser.add_argument('--logpath', type=str, default='log', help='Directory to load the model checkpoint')
    
    args = parser.parse_args()
    main(args)
