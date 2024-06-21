import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_data_loader(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # ResNet expects 224x224 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def get_test_loader(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # ResNet expects 224x224 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return test_loader
