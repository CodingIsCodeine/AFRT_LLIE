"""
Training Script for AFRT Model

This script handles the complete training pipeline:
- Loading configuration and dataset
- Building model and optimizer
- Training loop with validation
- Checkpoint saving
- Evaluation metrics logging

Author: AFRT Implementation
Date: 2026
"""

import os
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm
import time

# Import our modules
from models import AFRT
from losses import AFRTLoss
from utils import get_dataloaders, compute_metrics, save_image


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found!")
        print("Please ensure config.yaml exists in the current directory.")
        raise
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        raise


def setup_device(config):
    """Setup computation device (CUDA or CPU)"""
    device_name = config['training']['device']
    
    if device_name == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print("Using CPU (training will be slower)")
    
    return device


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """
    Train for one epoch
    
    Args:
        model: AFRT model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        epoch: Current epoch number
        total_epochs: Total number of epochs
    
    Returns:
        avg_loss: Average loss for the epoch
        avg_loss_dict: Dictionary of average component losses
    """
    model.train()
    
    total_loss = 0.0
    loss_components = {}
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{total_epochs}] Training")
    
    for batch_idx, (low_imgs, high_imgs) in enumerate(pbar):
        # Move to device
        low_imgs = low_imgs.to(device)
        high_imgs = high_imgs.to(device)
        
        # Forward pass
        enhanced_imgs, illum_map = model(low_imgs)
        
        # Compute loss
        loss, loss_dict = criterion(enhanced_imgs, high_imgs, illum_map)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for key, value in loss_dict.items():
            if key not in loss_components:
                loss_components[key] = 0.0
            loss_components[key] += value
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'l1': f"{loss_dict.get('l1', 0):.4f}"
        })
    
    # Compute averages
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_loss_dict = {k: v / num_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_loss_dict


def validate(model, test_loader, criterion, device):
    """
    Validate the model
    
    Args:
        model: AFRT model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to run on
    
    Returns:
        avg_loss: Average validation loss
        avg_psnr: Average PSNR
        avg_ssim: Average SSIM
    """
    model.eval()
    
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Validating")
        
        for low_imgs, high_imgs in pbar:
            # Move to device
            low_imgs = low_imgs.to(device)
            high_imgs = high_imgs.to(device)
            
            # Forward pass
            enhanced_imgs, illum_map = model(low_imgs)
            
            # Compute loss
            loss, _ = criterion(enhanced_imgs, high_imgs, illum_map)
            total_loss += loss.item()
            
            # Compute metrics
            metrics = compute_metrics(enhanced_imgs, high_imgs)
            total_psnr += metrics['psnr']
            total_ssim += metrics['ssim']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'psnr': f"{metrics['psnr']:.2f}",
                'ssim': f"{metrics['ssim']:.4f}"
            })
    
    # Compute averages
    num_batches = len(test_loader)
    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    return avg_loss, avg_psnr, avg_ssim


def save_checkpoint(model, optimizer, epoch, save_path, is_best=False):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
        
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")


def main():
    """Main training function"""
    print("=" * 80)
    print("AFRT Training Script")
    print("=" * 80)
    
    # Load configuration
    config = load_config('config.yaml')
    
    # Setup device
    device = setup_device(config)
    
    # Create checkpoint directory
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Build model
    print("\nBuilding model...")
    model = AFRT(
        dim=config['model']['dim'],
        num_blocks=config['model']['num_blocks']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Build loss function
    print("\nBuilding loss function...")
    criterion = AFRTLoss(weights=config.get('loss_weights', None)).to(device)
    
    # Build optimizer
    print("\nBuilding optimizer...")
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler (optional but recommended)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    # Load datasets
    print("\nLoading datasets...")
    try:
        train_loader, test_loader = get_dataloaders(
            dataset_path=config['training']['dataset_path'],
            batch_size=config['training']['batch_size'],
            image_size=256,
            num_workers=config['training']['num_workers']
        )
    except Exception as e:
        print(f"\nError loading dataset: {str(e)}")
        print("\nPlease ensure your dataset is organized as follows:")
        print(f"  {config['training']['dataset_path']}/train/low/")
        print(f"  {config['training']['dataset_path']}/train/high/")
        print(f"  {config['training']['dataset_path']}/test/low/")
        print(f"  {config['training']['dataset_path']}/test/high/")
        print("\nSee README.md for detailed dataset setup instructions.")
        return
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    total_epochs = config['training']['epochs']
    save_freq = config['training']['save_freq']
    best_psnr = 0.0
    
    start_time = time.time()
    
    for epoch in range(total_epochs):
        print(f"\n{'=' * 80}")
        print(f"Epoch [{epoch+1}/{total_epochs}]")
        print(f"{'=' * 80}")
        
        # Train
        avg_loss, loss_dict = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, total_epochs
        )
        
        print(f"\nTraining Results:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Loss Components:")
        for key, value in loss_dict.items():
            if key != 'total':
                print(f"    {key}: {value:.4f}")
        
        # Validate
        print("\nValidating...")
        val_loss, val_psnr, val_ssim = validate(model, test_loader, criterion, device)
        
        print(f"\nValidation Results:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  PSNR: {val_psnr:.2f} dB")
        print(f"  SSIM: {val_ssim:.4f}")
        
        # Update learning rate based on validation PSNR
        scheduler.step(val_psnr)
        
        # Save checkpoint every save_freq epochs
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
        
        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, best_checkpoint_path, is_best=True)
            print(f"  ★ New best model! PSNR: {best_psnr:.2f} dB")
        
        # Estimate time remaining
        elapsed_time = time.time() - start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        remaining_epochs = total_epochs - (epoch + 1)
        estimated_remaining = avg_time_per_epoch * remaining_epochs
        
        print(f"\nTime: {elapsed_time/3600:.2f}h elapsed, "
              f"{estimated_remaining/3600:.2f}h remaining (estimated)")
    
    # Training complete
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Total training time: {(time.time() - start_time)/3600:.2f} hours")
    print(f"Best validation PSNR: {best_psnr:.2f} dB")
    print(f"Best model saved to: {os.path.join(checkpoint_dir, 'best_model.pth')}")
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pth")
    save_checkpoint(model, optimizer, total_epochs-1, final_checkpoint_path)
    print(f"Final model saved to: {final_checkpoint_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()