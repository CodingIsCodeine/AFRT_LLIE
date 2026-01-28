"""
Loss Functions for AFRT Training

This module implements all loss functions used in AFRT training:
- L1 reconstruction loss
- Perceptual loss (VGG-based)
- SSIM loss
- Frequency-specific losses
- Total variation loss
- Color consistency loss

Author: AFRT Implementation
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained VGG19_bn features
    Compares feature maps from intermediate layers
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Load pretrained VGG19 with batch normalization
        vgg = models.vgg19_bn(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        # Extract features up to relu4_2 (good balance of high/low level features)
        # VGG19_bn layers: conv-bn-relu blocks
        # We want features after the 4th pooling layer
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:40]).eval()
        
        # Freeze VGG parameters (we don't train it)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # MSE loss for feature comparison
        self.criterion = nn.MSELoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
        
        Returns:
            loss: Perceptual loss value
        """
        # Normalize to ImageNet stats (VGG expects this)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        
        # Extract features
        pred_features = self.feature_extractor(pred_norm)
        target_features = self.feature_extractor(target_norm)
        
        # Compute MSE on features
        loss = self.criterion(pred_features, target_features)
        
        return loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss
    Measures structural similarity between images
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3  # RGB
        self.window = self._create_window(window_size, self.channel)
    
    def _gaussian(self, window_size, sigma):
        """Create 1D Gaussian kernel"""
        gauss = torch.tensor([
            math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size, channel):
        """Create 2D Gaussian window"""
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """Compute SSIM between two images"""
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
        
        Returns:
            loss: 1 - SSIM (so minimizing increases similarity)
        """
        # Move window to same device as input
        if self.window.device != pred.device:
            self.window = self.window.to(pred.device)
        
        ssim_value = self._ssim(pred, target, self.window, self.window_size, self.channel, self.size_average)
        
        # Return loss (1 - SSIM, so we minimize)
        return 1 - ssim_value


def total_variation_loss(img):
    """
    Total Variation Loss - encourages spatial smoothness
    
    Args:
        img: Image tensor [B, C, H, W]
    
    Returns:
        tv_loss: Total variation loss value
    """
    # Compute differences in x and y directions
    tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
    tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
    
    return tv_h + tv_w


def color_consistency_loss(pred, target):
    """
    Color Consistency Loss - preserves color distribution
    Compares normalized color channels
    
    Args:
        pred: Predicted image [B, 3, H, W]
        target: Target image [B, 3, H, W]
    
    Returns:
        cc_loss: Color consistency loss value
    """
    # Normalize each channel by its mean to focus on color distribution
    pred_mean = pred.mean(dim=[2, 3], keepdim=True) + 1e-6  # Avoid division by zero
    target_mean = target.mean(dim=[2, 3], keepdim=True) + 1e-6
    
    pred_normalized = pred / pred_mean
    target_normalized = target / target_mean
    
    # MSE on normalized colors
    cc_loss = F.mse_loss(pred_normalized, target_normalized)
    
    return cc_loss


class FrequencyDecomposition(nn.Module):
    """
    Frequency decomposition for frequency-specific losses
    (Same as in models.py, but kept here for loss computation)
    """
    def __init__(self, in_channels=3):
        super(FrequencyDecomposition, self).__init__()
        self.low_pass = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.high_pass = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2, bias=False)
        
        nn.init.kaiming_normal_(self.low_pass.weight)
        nn.init.kaiming_normal_(self.high_pass.weight)
    
    def forward(self, x):
        low_freq = self.low_pass(x)
        high_freq = self.high_pass(x)
        return low_freq, high_freq


class AFRTLoss(nn.Module):
    """
    Combined loss function for AFRT training
    
    Includes:
    - L1 reconstruction loss
    - Perceptual loss (VGG)
    - SSIM loss
    - Frequency-specific L1 losses (low and high)
    - Total variation on illumination map
    - Color consistency loss
    
    Args:
        weights: Dictionary of loss weights
    """
    def __init__(self, weights=None):
        super(AFRTLoss, self).__init__()
        
        # Default weights if not provided
        if weights is None:
            weights = {
                'l1': 1.0,
                'perceptual': 0.1,
                'ssim': 0.5,
                'freq_low': 0.3,
                'freq_high': 0.3,
                'tv': 0.01,
                'color': 0.1
            }
        
        self.weights = weights
        
        # Initialize loss components
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.ssim_loss = SSIMLoss()
        self.freq_decomp = FrequencyDecomposition(in_channels=3)
        
        print("AFRTLoss initialized with weights:")
        for key, value in self.weights.items():
            print(f"  {key}: {value}")
    
    def forward(self, pred, target, illum_map):
        """
        Compute total loss
        
        Args:
            pred: Predicted enhanced image [B, 3, H, W]
            target: Ground truth image [B, 3, H, W]
            illum_map: Estimated illumination map [B, 1, H, W]
        
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary of individual loss components for logging
        """
        loss_dict = {}
        
        # 1. L1 Reconstruction Loss
        loss_l1 = self.l1_loss(pred, target)
        loss_dict['l1'] = loss_l1.item()
        
        # 2. Perceptual Loss
        loss_perceptual = self.perceptual_loss(pred, target)
        loss_dict['perceptual'] = loss_perceptual.item()
        
        # 3. SSIM Loss
        loss_ssim = self.ssim_loss(pred, target)
        loss_dict['ssim'] = loss_ssim.item()
        
        # 4. Frequency-specific losses
        pred_low, pred_high = self.freq_decomp(pred)
        target_low, target_high = self.freq_decomp(target)
        
        loss_freq_low = self.l1_loss(pred_low, target_low)
        loss_freq_high = self.l1_loss(pred_high, target_high)
        loss_dict['freq_low'] = loss_freq_low.item()
        loss_dict['freq_high'] = loss_freq_high.item()
        
        # 5. Total Variation on illumination map (encourages smooth illumination)
        loss_tv = total_variation_loss(illum_map)
        loss_dict['tv'] = loss_tv.item()
        
        # 6. Color Consistency Loss
        loss_color = color_consistency_loss(pred, target)
        loss_dict['color'] = loss_color.item()
        
        # Combine all losses with weights
        total_loss = (
            self.weights['l1'] * loss_l1 +
            self.weights['perceptual'] * loss_perceptual +
            self.weights['ssim'] * loss_ssim +
            self.weights['freq_low'] * loss_freq_low +
            self.weights['freq_high'] * loss_freq_high +
            self.weights['tv'] * loss_tv +
            self.weights['color'] * loss_color
        )
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# Test function
if __name__ == "__main__":
    print("Testing Loss Functions...")
    
    # Create dummy data
    batch_size = 2
    pred = torch.randn(batch_size, 3, 256, 256)
    target = torch.randn(batch_size, 3, 256, 256)
    illum_map = torch.randn(batch_size, 1, 256, 256)
    
    # Test individual losses
    print("\n1. Testing L1 Loss...")
    l1_loss = nn.L1Loss()
    print(f"   L1 Loss: {l1_loss(pred, target).item():.4f}")
    
    print("\n2. Testing Perceptual Loss...")
    perceptual_loss = PerceptualLoss()
    print(f"   Perceptual Loss: {perceptual_loss(pred, target).item():.4f}")
    
    print("\n3. Testing SSIM Loss...")
    ssim_loss = SSIMLoss()
    print(f"   SSIM Loss: {ssim_loss(pred, target).item():.4f}")
    
    print("\n4. Testing Total Variation Loss...")
    tv_loss = total_variation_loss(illum_map)
    print(f"   TV Loss: {tv_loss.item():.4f}")
    
    print("\n5. Testing Color Consistency Loss...")
    cc_loss = color_consistency_loss(pred, target)
    print(f"   Color Consistency Loss: {cc_loss.item():.4f}")
    
    print("\n6. Testing Combined AFRT Loss...")
    afrt_loss = AFRTLoss()
    total_loss, loss_dict = afrt_loss(pred, target, illum_map)
    print(f"   Total Loss: {total_loss.item():.4f}")
    print("   Loss Components:")
    for key, value in loss_dict.items():
        print(f"     {key}: {value:.4f}")
    
    print("\n✓ All loss functions test passed!")