"""
Inference Script for AFRT Model

This script performs inference on low-light images using a trained AFRT model.
Supports both single image and batch processing.

Usage:
    Single image:
        python inference.py --input_image path/to/image.jpg --output_path outputs/enhanced.jpg
    
    Batch (folder):
        python inference.py --input_image path/to/folder/ --output_path outputs/

Author: AFRT Implementation
Date: 2026
"""

import os
import argparse
import yaml
import torch
import time
from glob import glob

# Import our modules
from models import AFRT
from utils import load_image, save_image


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Configuration file '{config_path}' not found!")
        print("Using default configuration...")
        return {
            'model': {'dim': 64, 'num_blocks': 6},
            'inference': {'model_path': 'checkpoints/best_model.pth', 'device': 'cuda'}
        }
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        raise


def load_model(model_path, device, dim=64, num_blocks=6):
    """
    Load trained AFRT model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        dim: Model dimension
        num_blocks: Number of transformer blocks
    
    Returns:
        model: Loaded AFRT model in eval mode
    """
    try:
        # Build model
        model = AFRT(dim=dim, num_blocks=num_blocks).to(device)
        
        # Load checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"  Model trained for {checkpoint['epoch']+1} epochs")
        else:
            # Assume checkpoint is just the state dict
            model.load_state_dict(checkpoint)
        
        # Set to eval mode
        model.eval()
        
        print("Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def enhance_image(model, image_path, device):
    """
    Enhance a single image
    
    Args:
        model: AFRT model
        image_path: Path to input image
        device: Device to run on
    
    Returns:
        enhanced_img: Enhanced image tensor [1, 3, H, W]
        original_size: Original image size (H, W)
    """
    # Load image
    img_tensor = load_image(image_path).to(device)
    original_size = img_tensor.shape[2:]  # (H, W)
    
    # Resize to multiple of 16 for better processing (optional)
    h, w = original_size
    h_new = ((h + 15) // 16) * 16
    w_new = ((w + 15) // 16) * 16
    
    if (h_new, w_new) != (h, w):
        img_tensor = torch.nn.functional.interpolate(
            img_tensor,
            size=(h_new, w_new),
            mode='bilinear',
            align_corners=True
        )
    
    # Enhance
    with torch.no_grad():
        enhanced_img, illum_map = model(img_tensor)
    
    # Resize back to original size
    if (h_new, w_new) != (h, w):
        enhanced_img = torch.nn.functional.interpolate(
            enhanced_img,
            size=original_size,
            mode='bilinear',
            align_corners=True
        )
    
    return enhanced_img, original_size


def process_single_image(args, model, device):
    """Process a single image"""
    print(f"\nProcessing image: {args.input_image}")
    
    start_time = time.time()
    
    # Enhance image
    enhanced_img, original_size = enhance_image(model, args.input_image, device)
    
    # Save result
    save_image(enhanced_img, args.output_path)
    
    elapsed_time = time.time() - start_time
    
    print(f"Enhanced image saved to: {args.output_path}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Image size: {original_size[0]}×{original_size[1]}")


def process_batch(args, model, device):
    """Process all images in a folder"""
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(args.input_image, ext)))
        image_files.extend(glob(os.path.join(args.input_image, ext.upper())))
    
    if len(image_files) == 0:
        print(f"No images found in {args.input_image}")
        return
    
    print(f"\nFound {len(image_files)} images to process")
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Process each image
    total_time = 0
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        
        start_time = time.time()
        
        # Enhance image
        enhanced_img, original_size = enhance_image(model, image_path, device)
        
        # Save with same filename
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_enhanced{ext}"
        output_path = os.path.join(args.output_path, output_filename)
        
        save_image(enhanced_img, output_path)
        
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        
        print(f"  Saved to: {output_filename}")
        print(f"  Time: {elapsed_time:.2f}s | Size: {original_size[0]}×{original_size[1]}")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("Batch Processing Complete!")
    print(f"{'=' * 80}")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/len(image_files):.2f} seconds")
    print(f"Output directory: {args.output_path}")


def main():
    """Main inference function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='AFRT Inference Script')
    parser.add_argument(
        '--input_image',
        type=str,
        required=True,
        help='Path to input image or folder containing images'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='outputs/enhanced.jpg',
        help='Path to save enhanced image(s)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to model checkpoint (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to run on (overrides config)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("AFRT Inference Script")
    print("=" * 80)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model_path is not None:
        config['inference']['model_path'] = args.model_path
    if args.device is not None:
        config['inference']['device'] = args.device
    
    # Setup device
    device_name = config['inference']['device']
    if device_name == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Warning: CUDA requested but not available. Using CPU.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load model
    try:
        model = load_model(
            model_path=config['inference']['model_path'],
            device=device,
            dim=config['model']['dim'],
            num_blocks=config['model']['num_blocks']
        )
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease ensure you have trained a model first using train.py")
        print("Or specify a valid model path with --model_path")
        return
    
    # Check if input is file or directory
    if os.path.isfile(args.input_image):
        # Single image mode
        process_single_image(args, model, device)
    elif os.path.isdir(args.input_image):
        # Batch mode
        process_batch(args, model, device)
    else:
        print(f"Error: Input path does not exist: {args.input_image}")
        return
    
    print("\n" + "=" * 80)
    print("Inference Complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInference interrupted by user.")
    except Exception as e:
        print(f"\n\nError during inference: {str(e)}")
        import traceback
        traceback.print_exc()