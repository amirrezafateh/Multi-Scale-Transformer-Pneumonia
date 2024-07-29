import argparse
import os
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from model.MSR import MSR
from data.data_loader_covid import get_test_loader

def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')

    # Compute AUC for each class and average
    all_targets_one_hot = np.eye(3)[all_targets]  # Convert to one-hot encoding
    all_preds_one_hot = np.eye(3)[all_preds]
    auc = roc_auc_score(all_targets_one_hot, all_preds_one_hot, average='macro', multi_class='ovo')

    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    return test_loss, test_acc, precision, recall, f1, auc, cm

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
    model = MSR(layers=args.layers, num_classes=3)  # 3-class classification

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
    test_loss, test_acc, precision, recall, f1, auc, cm = test(model, test_loader, criterion, device)

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}')
    print('Confusion Matrix:')
    print('                   Predicted')
    print('                 COVID19 NORMAL PNEUMONIA')
    print('Actual COVID19     {}      {}       {}'.format(cm[0, 0], cm[0, 1], cm[0, 2]))
    print('       NORMAL      {}      {}       {}'.format(cm[1, 0], cm[1, 1], cm[1, 2]))
    print('    PNEUMONIA      {}      {}       {}'.format(cm[2, 0], cm[2, 1], cm[2, 2]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MSR model with ResNet backbone on Pneumonia dataset")
    parser.add_argument('--bsz', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--layers', type=int, default=50, help='Number of layers in ResNet backbone (e.g., 50 or 101)')
    parser.add_argument('--data_dir', type=str, default='./data/chest_xray', help='Directory where dataset is located')
    parser.add_argument('--logpath', type=str, default='log', help='Directory to load the model checkpoint')
    
    args = parser.parse_args()
    main(args)
