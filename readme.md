# Adaptive Frequency-Aware Retinex Transformer (AFRT)

A complete PyTorch implementation for low-light image enhancement combining Retinex theory, frequency decomposition, and transformer attention.

## 📋 Table of Contents

- [Setup](#setup)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## 🚀 Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- Basic Python installation

### Step 1: Create Virtual Environment

**Windows:**
```bash
python -m venv afrt_env
afrt_env\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv afrt_env
source afrt_env/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Create Directory Structure

```bash
mkdir -p data/lol/train/low data/lol/train/high data/lol/test/low data/lol/test/high
mkdir -p checkpoints outputs
```

## 📂 Dataset Preparation

### LOL Dataset (Recommended)

1. Download the LOL dataset from: https://daooshee.github.io/BMVC2018website/
2. Extract the dataset
3. Organize files as follows:

```
data/lol/
├── train/
│   ├── low/     # Low-light training images
│   └── high/    # Normal-light training images (ground truth)
└── test/
    ├── low/     # Low-light test images
    └── high/    # Normal-light test images (ground truth)
```

**Important:** Ensure paired images have the same filename in `low/` and `high/` folders.

### Using Your Own Dataset

- Place paired low/normal light images in the structure above
- Images should be RGB (`.jpg`, `.png`)
- Recommended resolution: 256×256 or higher

## 🎓 Training

### Basic Training

```bash
python train.py
```

### Advanced Options

Edit `config.yaml` to customize:

- `batch_size`: Reduce if GPU runs out of memory (try 2 or 1)
- `epochs`: Number of training iterations (default: 100)
- `lr`: Learning rate (default: 0.001)
- `dim`: Model dimension (default: 64, increase for better quality)

### Monitor Training

The script will print:
- Loss values every batch
- Validation PSNR/SSIM every epoch
- Checkpoint saves every 10 epochs

```
Epoch [1/100], Batch [10/50], Loss: 0.3452
Epoch 1 Complete - Avg Loss: 0.3245, Val PSNR: 18.32, Val SSIM: 0.7234
Checkpoint saved: checkpoints/model_epoch_10.pth
```

### Training Tips

1. **Start Small**: Train for 10 epochs first to verify everything works
2. **GPU Memory**: If you get CUDA out of memory, reduce `batch_size` in config.yaml
3. **Training Time**: Expect ~2-5 hours for 100 epochs on a modern GPU
4. **Best Model**: The final model is saved as `checkpoints/best_model.pth`

## 🖼️ Inference

### Single Image

```bash
python inference.py --input_image path/to/dark_image.jpg --output_path outputs/enhanced.jpg
```

### Batch Processing (Multiple Images)

```bash
python inference.py --input_image path/to/folder/ --output_path outputs/
```

### Example

```bash
# Enhance a single low-light photo
python inference.py --input_image data/lol/test/low/001.png --output_path outputs/001_enhanced.png

# Process all images in a folder
python inference.py --input_image data/lol/test/low/ --output_path outputs/
```

## 📁 Project Structure

```
afrt_project/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── train.py                 # Training script
├── inference.py             # Inference script
├── models.py                # AFRT model architecture
├── losses.py                # Loss functions
├── utils.py                 # Utilities (data loading, metrics)
├── data/                    # Dataset directory
│   └── lol/
│       ├── train/
│       └── test/
├── checkpoints/             # Saved model weights
└── outputs/                 # Enhanced images
```

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
- Reduce `batch_size` in `config.yaml` (try 2 or 1)
- Reduce `dim` in `config.yaml` (try 32)
- Close other GPU applications

#### 2. No Module Named 'yaml'

**Error:** `ModuleNotFoundError: No module named 'yaml'`

**Solution:**
```bash
pip install pyyaml
```

#### 3. Dataset Not Found

**Error:** `FileNotFoundError: [Errno 2] No such file or directory`

**Solution:**
- Verify dataset is in `data/lol/train/` and `data/lol/test/`
- Check that both `low/` and `high/` folders exist
- Ensure paired images have matching filenames

#### 4. Model Not Converging

**Symptoms:** Loss not decreasing, poor validation PSNR

**Solutions:**
- Verify dataset quality (check that low/high pairs match)
- Reduce learning rate in `config.yaml` (try 0.0001)
- Increase training epochs
- Check that images are properly normalized [0, 1]

#### 5. CPU vs GPU

**By default, the code uses GPU if available:**
- To force CPU: Change `device: 'cpu'` in `config.yaml`
- To check GPU availability:
```python
import torch
print(torch.cuda.is_available())
```

### Performance Tips

1. **First-Time Setup**: Run for 5 epochs to verify installation
2. **Quality vs Speed**: Higher `dim` = better quality but slower
3. **Checkpoints**: Keep every 10th epoch checkpoint for comparison
4. **Validation**: Watch validation PSNR - should increase over epochs

## 📊 Expected Results

After 100 epochs on LOL dataset:
- **Validation PSNR**: 20-24 dB
- **Validation SSIM**: 0.80-0.88
- **Training Time**: 2-5 hours (GPU dependent)

## 🆘 Getting Help

1. Check this README's troubleshooting section
2. Verify all files are present and correctly named
3. Ensure virtual environment is activated
4. Check that dataset structure matches exactly

## 📝 Notes

- **GPU Recommended**: Training on CPU will be 10-20× slower
- **Disk Space**: ~2GB for model checkpoints, adjust save frequency in train.py if needed
- **Customization**: All hyperparameters in `config.yaml` can be tuned
- **Pretrained Models**: After training, share `checkpoints/best_model.pth` for others to use

## 🎯 Quick Start Checklist

- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset downloaded and organized in `data/lol/`
- [ ] Directory structure created (`mkdir checkpoints outputs`)
- [ ] Config file reviewed (`config.yaml`)
- [ ] Training started (`python train.py`)
- [ ] Model checkpoints saving to `checkpoints/`
- [ ] Inference tested on sample image
