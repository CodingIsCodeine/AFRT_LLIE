"""
AFRT (Adaptive Frequency-Aware Retinex Transformer) Model Architecture

This module implements all components of the AFRT model for low-light image enhancement.
The architecture combines frequency decomposition, Retinex theory, and transformer attention.

Author: AFRT Implementation
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv2d -> BatchNorm -> ReLU
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolutional kernel
        stride: Stride for convolution
        padding: Padding for convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    Residual block with two ConvBlocks and skip connection
    
    Args:
        channels: Number of input/output channels
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual  # Skip connection


class FrequencyDecomposition(nn.Module):
    """
    Learnable frequency decomposition into low and high frequency components
    
    Low-freq captures illumination and structure
    High-freq captures details and noise
    
    Args:
        in_channels: Number of input channels (typically 3 for RGB)
    """
    def __init__(self, in_channels=3):
        super(FrequencyDecomposition, self).__init__()
        # Learnable low-pass filter (captures smooth illumination)
        self.low_pass = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2, bias=False)
        # Learnable high-pass filter (captures details)
        self.high_pass = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2, bias=False)
        
        # Initialize with Kaiming initialization for better training
        nn.init.kaiming_normal_(self.low_pass.weight)
        nn.init.kaiming_normal_(self.high_pass.weight)
    
    def forward(self, x):
        """
        Args:
            x: Input image [B, 3, H, W]
        
        Returns:
            low_freq: Low frequency component [B, 3, H, W]
            high_freq: High frequency component [B, 3, H, W]
        """
        low_freq = self.low_pass(x)
        high_freq = self.high_pass(x)
        return low_freq, high_freq


class IlluminationEstimator(nn.Module):
    """
    U-Net style encoder-decoder to estimate illumination map from low-frequency component
    Includes global context via global average pooling and FC layers
    
    Args:
        in_channels: Number of input channels
        dim: Base feature dimension
    """
    def __init__(self, in_channels=3, dim=64):
        super(IlluminationEstimator, self).__init__()
        
        # Encoder pathway (downsampling)
        self.enc1 = ConvBlock(in_channels, dim)
        self.enc2 = ConvBlock(dim, dim * 2, stride=2)  # Downsample
        self.enc3 = ConvBlock(dim * 2, dim * 4, stride=2)  # Downsample
        
        # Global context pathway
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(dim * 4, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim * 4)
        self.relu = nn.ReLU(inplace=True)
        
        # Bottleneck
        self.bottleneck = ConvBlock(dim * 4, dim * 4)
        
        # Decoder pathway (upsampling with skip connections)
        self.dec3 = ConvBlock(dim * 6, dim * 2)  # dim*4 from bottleneck + dim*4 from skip
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dec2 = ConvBlock(dim * 3, dim)  # ← FIXED: dim*2 from dec3 + dim from skip = dim*3
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dec1 = ConvBlock(dim * 2, dim)  # dim from dec2 + dim from skip
        
        # Output projection to illumination map [0, 1]
        self.output = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Constrain to [0, 1]
        )
    
    def forward(self, x):
        """
        Args:
            x: Low frequency input [B, 3, H, W]
        
        Returns:
            illum_map: Estimated illumination [B, 1, H, W] in range [0, 1]
        """
        # Encoder with skip connections
        e1 = self.enc1(x)  # [B, dim, H, W]
        e2 = self.enc2(e1)  # [B, dim*2, H/2, W/2]
        e3 = self.enc3(e2)  # [B, dim*4, H/4, W/4]
        
        # Global context
        B, C, H, W = e3.shape
        context = self.global_pool(e3).view(B, C)  # [B, dim*4]
        context = self.relu(self.fc1(context))  # [B, dim*2]
        context = self.fc2(context).view(B, C, 1, 1)  # [B, dim*4, 1, 1]
        
        # Bottleneck with context
        bottleneck = self.bottleneck(e3 + context)  # [B, dim*4, H/4, W/4]
        
        # Decoder with skip connections
        d3 = self.up3(bottleneck)  # [B, dim*4, H/2, W/2]
        d3 = torch.cat([d3, e2], dim=1)  # [B, dim*8, H/2, W/2]
        d3 = self.dec3(d3)  # [B, dim*2, H/2, W/2]
        
        d2 = self.up2(d3)  # [B, dim*2, H, W]
        d2 = torch.cat([d2, e1], dim=1)  # [B, dim*3, H, W] ← This is 192 channels with dim=64
        d2 = self.dec2(d2)  # [B, dim, H, W]
        
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # [B, dim, H, W]
        
        # Output illumination map
        illum_map = self.output(d1)  # [B, 1, H, W]
        return illum_map


class DetailNet(nn.Module):
    """
    Network to process high-frequency details
    Uses residual blocks to preserve fine details
    
    Args:
        in_channels: Number of input channels
        dim: Feature dimension
    """
    def __init__(self, in_channels=3, dim=64):
        super(DetailNet, self).__init__()
        self.conv_in = ConvBlock(in_channels, dim)
        self.res1 = ResidualBlock(dim)
        self.res2 = ResidualBlock(dim)
        self.conv_out = ConvBlock(dim, dim)
    
    def forward(self, x):
        """
        Args:
            x: High frequency input [B, 3, H, W]
        
        Returns:
            detail_features: Processed details [B, dim, H, W]
        """
        x = self.conv_in(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.conv_out(x)
        return x


class AdaptiveIGAttention(nn.Module):
    """
    Adaptive Illumination-Guided Attention Block
    
    Combines self-attention with illumination guidance and learned level embeddings
    
    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        num_levels: Number of illumination levels to embed
    """
    def __init__(self, dim, num_heads=8, num_levels=10):
        super(AdaptiveIGAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Q, K, V projections for self-attention
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj_out = nn.Linear(dim, dim)
        
        # Illumination gating mechanism
        self.illum_gate_conv = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Learnable illumination level embeddings
        self.num_levels = num_levels
        self.level_embeddings = nn.Parameter(torch.randn(num_levels, dim))
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, illum_map):
        """
        Args:
            x: Input features [B, dim, H, W]
            illum_map: Illumination map [B, 1, H, W]
        
        Returns:
            out: Attention output [B, dim, H, W]
        """
        B, C, H, W = x.shape
        
        # Prepare for attention: reshape to [B, H*W, C]
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x_norm = self.norm(x_flat)
        
        # Compute Q, K, V
        qkv = self.qkv(x_norm).reshape(B, H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, H*W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, num_heads, H*W, H*W]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)  # [B, H*W, C]
        out = self.proj_out(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        
        # Illumination gating
        gate = self.illum_gate_conv(illum_map)  # [B, dim, H, W]
        out = out * gate
        
        # Add illumination level embeddings
        # Quantize illumination to discrete levels
        illum_levels = (illum_map * (self.num_levels - 1)).long().clamp(0, self.num_levels - 1)  # [B, 1, H, W]
        illum_levels = illum_levels.squeeze(1)  # [B, H, W]
        
        # Gather embeddings for each pixel
        level_embed = self.level_embeddings[illum_levels]  # [B, H, W, dim]
        level_embed = level_embed.permute(0, 3, 1, 2)  # [B, dim, H, W]
        
        # Adaptive addition
        out = out + level_embed
        
        return out


class FrequencyAdaptiveFusion(nn.Module):
    """
    Frequency-adaptive fusion of low and high frequency features
    Uses illumination map to determine fusion weights
    
    Args:
        dim: Feature dimension
    """
    def __init__(self, dim):
        super(FrequencyAdaptiveFusion, self).__init__()
        # Network to predict fusion weights from illumination
        self.fusion_net = nn.Sequential(
            nn.Conv2d(1, dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 2, kernel_size=3, padding=1),
            nn.Softmax(dim=1)  # Normalize weights across 2 channels
        )
        
        # Process low and high features before fusion
        self.low_conv = ConvBlock(dim, dim)
        self.high_conv = ConvBlock(dim, dim)
    
    def forward(self, low_features, high_features, illum_map):
        """
        Args:
            low_features: Low frequency features [B, dim, H, W]
            high_features: High frequency features [B, dim, H, W]
            illum_map: Illumination map [B, 1, H, W]
        
        Returns:
            fused: Fused features [B, dim, H, W]
        """
        # Compute adaptive fusion weights
        weights = self.fusion_net(illum_map)  # [B, 2, H, W]
        alpha = weights[:, 0:1, :, :]  # [B, 1, H, W]
        beta = weights[:, 1:2, :, :]  # [B, 1, H, W]
        
        # Process features
        low_proc = self.low_conv(low_features)
        high_proc = self.high_conv(high_features)
        
        # Weighted fusion
        fused = alpha * low_proc + beta * high_proc
        return fused


class MultiScaleRefinement(nn.Module):
    """
    Multi-scale feature refinement at different resolutions
    
    Args:
        dim: Feature dimension
        scales: List of scales to process (e.g., [1.0, 0.5, 0.25])
    """
    def __init__(self, dim, scales=[1.0, 0.5, 0.25]):
        super(MultiScaleRefinement, self).__init__()
        self.scales = scales
        
        # Refinement blocks for each scale
        self.refiners = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(dim),
                ResidualBlock(dim)
            ) for _ in scales
        ])
    
    def forward(self, x):
        """
        Args:
            x: Input features [B, dim, H, W]
        
        Returns:
            refined: Multi-scale refined features [B, dim, H, W]
        """
        B, C, H, W = x.shape
        outputs = []
        
        for scale, refiner in zip(self.scales, self.refiners):
            if scale != 1.0:
                # Downsample
                h_new, w_new = int(H * scale), int(W * scale)
                x_scaled = F.interpolate(x, size=(h_new, w_new), mode='bilinear', align_corners=True)
            else:
                x_scaled = x
            
            # Refine at this scale
            refined = refiner(x_scaled)
            
            # Upsample back to original size
            if scale != 1.0:
                refined = F.interpolate(refined, size=(H, W), mode='bilinear', align_corners=True)
            
            outputs.append(refined)
        
        # Average all scales
        refined = torch.stack(outputs, dim=0).mean(dim=0)
        return refined


class NoiseSuppression(nn.Module):
    """
    Noise suppression module using separable convolution and channel attention
    
    Args:
        dim: Feature dimension
    """
    def __init__(self, dim):
        super(NoiseSuppression, self).__init__()
        # Separable convolution (depthwise + pointwise)
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        
        # Channel attention
        reduction_ratio = 16
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction_ratio, dim, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: Input features [B, dim, H, W]
        
        Returns:
            out: Noise-suppressed features [B, dim, H, W]
        """
        # Separable convolution
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.relu(out)
        
        # Channel attention
        attn = self.channel_attn(out)
        out = out * attn
        
        return out


class ColorRestoration(nn.Module):
    """
    Color restoration module to correct color cast
    Uses learnable 3x3 color transformation matrix
    
    Args:
        dim: Feature dimension
    """
    def __init__(self, dim):
        super(ColorRestoration, self).__init__()
        self.conv = ConvBlock(dim, 3)
        
        # Learnable 3x3 color transformation matrix
        self.color_matrix = nn.Parameter(torch.eye(3))
    
    def forward(self, x):
        """
        Args:
            x: Input features [B, dim, H, W]
        
        Returns:
            out: Color-corrected RGB [B, 3, H, W]
        """
        # Project to 3 channels
        out = self.conv(x)  # [B, 3, H, W]
        
        # Apply color transformation matrix
        B, C, H, W = out.shape
        out_flat = out.view(B, 3, -1)  # [B, 3, H*W]
        out_transformed = torch.matmul(self.color_matrix, out_flat)  # [B, 3, H*W]
        out = out_transformed.view(B, 3, H, W)  # [B, 3, H, W]
        
        return out


class AFRT(nn.Module):
    """
    Adaptive Frequency-Aware Retinex Transformer
    
    Complete model for low-light image enhancement combining:
    - Frequency decomposition
    - Illumination estimation
    - Transformer attention
    - Multi-scale refinement
    - Noise suppression and color restoration
    
    Args:
        dim: Base feature dimension (default: 64)
        num_blocks: Number of transformer blocks (default: 6)
    """
    def __init__(self, dim=64, num_blocks=6):
        super(AFRT, self).__init__()
        
        # 1. Frequency Decomposition
        self.freq_decomp = FrequencyDecomposition(in_channels=3)
        
        # 2. Low-Frequency Branch: Illumination Estimation
        self.illum_estimator = IlluminationEstimator(in_channels=3, dim=dim)
        
        # 3. High-Frequency Branch: Detail Processing
        self.detail_net = DetailNet(in_channels=3, dim=dim)
        
        # 4. Project low-freq to feature space
        self.low_freq_proj = ConvBlock(3, dim)
        
        # 5. Transformer Blocks with Illumination Guidance
        self.transformer_blocks = nn.ModuleList([
            AdaptiveIGAttention(dim=dim, num_heads=8, num_levels=10)
            for _ in range(num_blocks)
        ])
        
        # 6. Frequency-Adaptive Fusion
        self.fusion = FrequencyAdaptiveFusion(dim=dim)
        
        # 7. Multi-Scale Refinement
        self.refinement = MultiScaleRefinement(dim=dim, scales=[1.0, 0.5, 0.25])
        
        # 8. Noise Suppression
        self.noise_suppression = NoiseSuppression(dim=dim)
        
        # 9. Color Restoration
        self.color_restoration = ColorRestoration(dim=dim)
    
    def forward(self, x):
        """
        Forward pass of AFRT
        
        Args:
            x: Input low-light image [B, 3, H, W] in range [0, 1]
        
        Returns:
            output: Enhanced image [B, 3, H, W] in range [0, 1]
            illum_map: Estimated illumination [B, 1, H, W] in range [0, 1]
        """
        # Ensure input is in correct range
        x = torch.clamp(x, 0.0, 1.0)
        
        # 1. Frequency Decomposition
        low_freq, high_freq = self.freq_decomp(x)  # [B, 3, H, W] each
        
        # 2. Illumination Estimation from low-frequency
        illum_map = self.illum_estimator(low_freq)  # [B, 1, H, W]
        
        # 3. Process high-frequency details
        detail_features = self.detail_net(high_freq)  # [B, dim, H, W]
        
        # 4. Project low-frequency to feature space
        low_features = self.low_freq_proj(low_freq)  # [B, dim, H, W]
        
        # 5. Apply Transformer Blocks to low features
        for block in self.transformer_blocks:
            low_features = low_features + block(low_features, illum_map)  # Residual connection
        
        # 6. Fuse low and high frequency features
        fused_features = self.fusion(low_features, detail_features, illum_map)  # [B, dim, H, W]
        
        # 7. Multi-scale refinement
        refined_features = self.refinement(fused_features)  # [B, dim, H, W]
        
        # 8. Noise suppression
        clean_features = self.noise_suppression(refined_features)  # [B, dim, H, W]
        
        # 9. Color restoration
        enhanced = self.color_restoration(clean_features)  # [B, 3, H, W]
        
        # Residual connection with input
        output = enhanced + x
        
        # Clamp to valid range [0, 1]
        output = torch.clamp(output, 0.0, 1.0)
        
        return output, illum_map


# Test function to verify model architecture
# if __name__ == "__main__":
#     print("Testing AFRT Model Architecture...")
    
#     # Create model
#     model = AFRT(dim=64, num_blocks=6)
    
#     # Count parameters
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Total parameters: {total_params:,}")
    
#     # Test 1: Small resolution for architecture verification
#     print("\n[Test 1] Architecture test with 64x64 input...")
#     test_input_small = torch.randn(2, 3, 64, 64)
#     with torch.no_grad():
#         output_small, illum_small = model(test_input_small)
#     print(f"  ✓ Small input passed: {test_input_small.shape} -> {output_small.shape}")
    
#     # Test 2: Full resolution with batch=1
#     print("\n[Test 2] Full resolution test with 256x256 input...")
#     test_input_full = torch.randn(1, 3, 256, 256)  # Batch size = 1
#     with torch.no_grad():
#         output_full, illum_full = model(test_input_full)
#     print(f"  ✓ Full input passed: {test_input_full.shape} -> {output_full.shape}")
    
#     # Verify output range
#     print(f"\nOutput range: [{output_full.min().item():.4f}, {output_full.max().item():.4f}]")
#     print(f"Illumination range: [{illum_full.min().item():.4f}, {illum_full.max().item():.4f}]")
    
#     print("\n✓ Model architecture test passed!")

if __name__ == "__main__":
    print("Testing AFRT Model Architecture...")
    
    # Use CPU for testing
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create small model for CPU testing
    model = AFRT(dim=32, num_blocks=2).to(device)  # Smaller for CPU
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test with small resolution (CPU can't handle large attention)
    print("\nTesting with 64x64 input (CPU compatible)...")
    test_input = torch.randn(1, 3, 64, 64).to(device)  # Batch=1, 64x64
    
    with torch.no_grad():
        output, illum_map = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Illumination shape: {illum_map.shape}")
    print(f"\nOutput range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"Illumination range: [{illum_map.min().item():.4f}, {illum_map.max().item():.4f}]")
    
    print("\n✓ Model architecture test passed!")
