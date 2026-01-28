"""
Utility Functions for AFRT

This module provides:
- Data loading for paired low-light/normal-light images
- Image save/load functions
- Evaluation metrics (PSNR, SSIM)
- Helper functions

Author: AFRT Implementation
Date: 2026
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


class LowLightDataset(Dataset):
    """
    Dataset for paired low-light and normal-light images
    
    Expected directory structure:
    dataset_path/
        train/
            low/    # Low-light images
            high/   # Normal-light images (ground truth)
        test/
            low/
            high/
    
    Args:
        dataset_path: Root path to dataset
        split: 'train' or 'test'
        image_size: Size to resize images (default: 256)
        augment: Whether to apply data augmentation (default: True for training)
    """
    def __init__(self, dataset_path, split='train', image_size=256, augment=True):
        self.dataset_path = dataset_path
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        
        # Paths to low and high directories
        self.low_dir = os.path.join(dataset_path, split, 'low')
        self.high_dir = os.path.join(dataset_path, split, 'high')
        
        # Check if directories exist
        if not os.path.exists(self.low_dir):
            raise FileNotFoundError(f"Low-light directory not found: {self.low_dir}")
        if not os.path.exists(self.high_dir):
            raise FileNotFoundError(f"Normal-light directory not found: {self.high_dir}")
        
        # Get list of image files
        self.low_images = sorted([f for f in os.listdir(self.low_dir) 
                                  if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        if len(self.low_images) == 0:
            raise ValueError(f"No images found in {self.low_dir}")
        
        print(f"Loaded {len(self.low_images)} image pairs for {split} split")
        
        # Define transforms
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.low_images)
    
    def __getitem__(self, idx):
        """
        Get a pair of low-light and normal-light images
        
        Returns:
            low_img: Low-light image tensor [3, H, W] normalized to [0, 1]
            high_img: Normal-light image tensor [3, H, W] normalized to [0, 1]
        """
        # Get image filename
        img_name = self.low_images[idx]
        
        # Load images
        low_path = os.path.join(self.low_dir, img_name)
        high_path = os.path.join(self.high_dir, img_name)
        
        try:
            # Read with OpenCV (BGR format)
            low_img = cv2.imread(low_path)
            high_img = cv2.imread(high_path)
            
            if low_img is None:
                raise ValueError(f"Failed to load low-light image: {low_path}")
            if high_img is None:
                raise ValueError(f"Failed to load normal-light image: {high_path}")
            
            # Convert BGR to RGB
            low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
            high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            low_img = cv2.resize(low_img, (self.image_size, self.image_size))
            high_img = cv2.resize(high_img, (self.image_size, self.image_size))
            
            # Apply augmentation if enabled
            if self.augment:
                # Random horizontal flip
                if np.random.rand() > 0.5:
                    low_img = cv2.flip(low_img, 1)
                    high_img = cv2.flip(high_img, 1)
                
                # Random rotation (0, 90, 180, 270 degrees)
                if np.random.rand() > 0.5:
                    k = np.random.randint(1, 4)  # 1, 2, or 3 (90, 180, or 270 degrees)
                    low_img = np.rot90(low_img, k)
                    high_img = np.rot90(high_img, k)
            
            # Convert to tensors and normalize to [0, 1]
            low_img = self.to_tensor(low_img.copy())  # Copy to ensure contiguous
            high_img = self.to_tensor(high_img.copy())
            
            return low_img, high_img
            
        except Exception as e:
            print(f"Error loading image pair {img_name}: {str(e)}")
            # Return a random valid sample instead
            return self.__getitem__((idx + 1) % len(self))


def get_dataloaders(dataset_path, batch_size=4, image_size=256, num_workers=4):
    """
    Create train and test dataloaders
    
    Args:
        dataset_path: Root path to dataset
        batch_size: Batch size for dataloaders
        image_size: Size to resize images
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
    """
    try:
        # Create datasets
        train_dataset = LowLightDataset(
            dataset_path=dataset_path,
            split='train',
            image_size=image_size,
            augment=True
        )
        
        test_dataset = LowLightDataset(
            dataset_path=dataset_path,
            split='test',
            image_size=image_size,
            augment=False
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True  # Drop last incomplete batch
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Use batch size 1 for testing
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"\nDataloaders created:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        return train_loader, test_loader
        
    except Exception as e:
        print(f"Error creating dataloaders: {str(e)}")
        print("Please check that your dataset is properly organized:")
        print(f"  {dataset_path}/train/low/")
        print(f"  {dataset_path}/train/high/")
        print(f"  {dataset_path}/test/low/")
        print(f"  {dataset_path}/test/high/")
        raise


def save_image(tensor, save_path):
    """
    Save a tensor as an image file
    
    Args:
        tensor: Image tensor [C, H, W] or [B, C, H, W] in range [0, 1]
        save_path: Path to save the image
    """
    try:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Handle batch dimension
        if tensor.dim() == 4:
            tensor = tensor[0]  # Take first image in batch
        
        # Clamp to [0, 1] and convert to numpy
        tensor = torch.clamp(tensor, 0.0, 1.0)
        img = tensor.cpu().detach().numpy()
        
        # Convert from [C, H, W] to [H, W, C]
        img = np.transpose(img, (1, 2, 0))
        
        # Convert to [0, 255] and uint8
        img = (img * 255.0).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Save
        cv2.imwrite(save_path, img)
        
    except Exception as e:
        print(f"Error saving image to {save_path}: {str(e)}")
        raise


def load_image(image_path, image_size=None):
    """
    Load an image as a PyTorch tensor
    
    Args:
        image_path: Path to image file
        image_size: Optional size to resize to
    
    Returns:
        img_tensor: Image tensor [1, 3, H, W] normalized to [0, 1]
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if specified
        if image_size is not None:
            img = cv2.resize(img, (image_size, image_size))
        
        # Convert to tensor [1, 3, H, W]
        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(img).unsqueeze(0)
        
        return img_tensor
        
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        raise


def compute_psnr(pred, target):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        pred: Predicted image tensor [B, C, H, W] or numpy array
        target: Target image tensor [B, C, H, W] or numpy array
    
    Returns:
        psnr: PSNR value in dB
    """
    # Convert tensors to numpy if needed
    if torch.is_tensor(pred):
        pred = pred.cpu().detach().numpy()
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    
    # Handle batch dimension
    if pred.ndim == 4:
        pred = pred[0]
        target = target[0]
    
    # Transpose from [C, H, W] to [H, W, C] if needed
    if pred.shape[0] == 3:
        pred = np.transpose(pred, (1, 2, 0))
        target = np.transpose(target, (1, 2, 0))
    
    # Compute PSNR
    psnr = compare_psnr(target, pred, data_range=1.0)
    
    return psnr


def compute_ssim(pred, target):
    """
    Compute Structural Similarity Index (SSIM)
    
    Args:
        pred: Predicted image tensor [B, C, H, W] or numpy array
        target: Target image tensor [B, C, H, W] or numpy array
    
    Returns:
        ssim: SSIM value in range [0, 1]
    """
    # Convert tensors to numpy if needed
    if torch.is_tensor(pred):
        pred = pred.cpu().detach().numpy()
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    
    # Handle batch dimension
    if pred.ndim == 4:
        pred = pred[0]
        target = target[0]
    
    # Transpose from [C, H, W] to [H, W, C] if needed
    if pred.shape[0] == 3:
        pred = np.transpose(pred, (1, 2, 0))
        target = np.transpose(target, (1, 2, 0))
    
    # Compute SSIM (channel_axis=2 for RGB)
    ssim = compare_ssim(target, pred, data_range=1.0, channel_axis=2)
    
    return ssim


def compute_metrics(pred, target):
    """
    Compute both PSNR and SSIM
    
    Args:
        pred: Predicted image tensor [B, C, H, W]
        target: Target image tensor [B, C, H, W]
    
    Returns:
        metrics: Dictionary with 'psnr' and 'ssim' keys
    """
    psnr = compute_psnr(pred, target)
    ssim = compute_ssim(pred, target)
    
    return {'psnr': psnr, 'ssim': ssim}


# Test function
if __name__ == "__main__":
    print("Testing Utility Functions...")
    
    # Test image save/load
    print("\n1. Testing save_image and load_image...")
    test_tensor = torch.rand(1, 3, 256, 256)
    test_path = "test_output.png"
    
    save_image(test_tensor, test_path)
    print(f"   Saved test image to {test_path}")
    
    loaded = load_image(test_path)
    print(f"   Loaded image shape: {loaded.shape}")
    
    # Clean up
    if os.path.exists(test_path):
        os.remove(test_path)
        print("   Cleaned up test image")
    
    # Test metrics
    print("\n2. Testing PSNR and SSIM...")
    pred = torch.rand(1, 3, 256, 256)
    target = torch.rand(1, 3, 256, 256)
    
    metrics = compute_metrics(pred, target)
    print(f"   PSNR: {metrics['psnr']:.2f} dB")
    print(f"   SSIM: {metrics['ssim']:.4f}")
    
    # Test with identical images (should give perfect scores)
    metrics_perfect = compute_metrics(pred, pred)
    print(f"   Perfect match PSNR: {metrics_perfect['psnr']:.2f} dB")
    print(f"   Perfect match SSIM: {metrics_perfect['ssim']:.4f}")
    
    print("\n✓ Utility functions test passed!")
    print("\nNote: To test dataloaders, you need a properly organized dataset.")
    print("Expected structure:")
    print("  data/lol/train/low/")
    print("  data/lol/train/high/")
    print("  data/lol/test/low/")
    print("  data/lol/test/high/")