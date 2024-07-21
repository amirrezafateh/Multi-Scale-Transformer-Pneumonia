import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from data.data_loader_segmentation import get_test_loader
from model.UNet import UNet

def generate_masks(test_loader, model, output_dir):
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = (outputs > 0.5).float()  # Assuming threshold of 0.5 for binary prediction

            for j in range(inputs.size(0)):
                pred_mask = predictions[j].cpu().numpy().squeeze()
                image_name = test_loader.dataset.images[i * test_loader.batch_size + j]
                save_path = os.path.join(output_dir, image_name.replace(".jpg", "_mask.png"))

                # Save predicted mask
                pred_mask = (pred_mask * 255).astype(np.uint8)
                Image.fromarray(pred_mask).save(save_path)

                print(f'Saved {save_path}')

def main(args):
    # Load test data
    test_loader = get_test_loader(args.data_dir, args.bsz)

    # Instantiate the UNet model
    model = UNet(in_channels=1, out_channels=1, feature_dims=64)  # Adjust based on your UNet definition
    model.load_state_dict(torch.load(args.model_path))  # Load trained model weights
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate masks using the trained model
    generate_masks(test_loader, model, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate masks using trained UNet model")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save predicted masks')
    parser.add_argument('--bsz', type=int, default=8, help='Batch size for testing')
    
    args = parser.parse_args()
    main(args)
