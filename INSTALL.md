# Installation Guide

## Quick Start Options

DisgraPhi can be used in three ways:

### 1. Google Colab (Recommended for Beginners)

**No installation needed!** Just click and start training:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/velocitatem/disgraPhi/blob/main/DisgraPhi_Personalization.ipynb)

See the [Colab Guide](COLAB_GUIDE.md) for detailed instructions.

### 2. PyPI Installation (Coming Soon)

```bash
pip install disgraphi
```

### 3. Development Installation (Current)

For the latest features and development:

```bash
# Clone repository
git clone https://github.com/velocitatem/disgraPhi.git
cd disgraPhi

# Option A: Use Makefile (recommended)
make venv
source .venv/bin/activate

# Option B: Manual setup
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .

# Option C: Install with all extras
pip install -e ".[all]"
```

## Installation Options

### Core Package (Minimum)
```bash
pip install -e .
```
Includes: Training, inference, data processing

### With Development Tools
```bash
pip install -e ".[dev]"
```
Adds: pytest, black, flake8, mypy

### With Inference Server
```bash
pip install -e ".[inference]"
```
Adds: FastAPI, uvicorn

### With Web Apps
```bash
pip install -e ".[webapp]"
```
Adds: Streamlit, Flask

### With Jupyter Notebooks
```bash
pip install -e ".[notebooks]"
```
Adds: Jupyter, ipywidgets, matplotlib

### Complete Installation
```bash
pip install -e ".[all]"
```
Installs everything!

## System Requirements

### Minimum Requirements
- **Python**: 3.10 or higher
- **RAM**: 8GB
- **Storage**: 10GB free space
- **OS**: Linux, macOS, or Windows with WSL

### Recommended for Training
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 16GB
- **Storage**: 20GB free space
- **CUDA**: 11.8 or higher

### Google Colab (Free Tier)
- **GPU**: T4 (15GB VRAM) - sufficient for training
- **RAM**: 12GB
- **Storage**: 78GB

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-pip
sudo apt-get install -y libzbar0  # For QR code processing

# Install DisgraPhi
git clone https://github.com/velocitatem/disgraPhi.git
cd disgraPhi
make venv
source .venv/bin/activate
```

### macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install python@3.10 zbar

# Install DisgraPhi
git clone https://github.com/velocitatem/disgraPhi.git
cd disgraPhi
make venv
source .venv/bin/activate
```

### Windows (WSL2)

1. Install WSL2 and Ubuntu:
```powershell
wsl --install -d Ubuntu-22.04
```

2. Inside WSL2, follow Linux instructions above

## GPU Setup

### NVIDIA GPU (CUDA)

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### AMD GPU (ROCm) - Experimental

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

### Apple Silicon (MPS)

PyTorch automatically uses Metal Performance Shaders:
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Verification

After installation, verify everything works:

```bash
# Check DisgraPhi installation
python -c "import ml.models.train; print('✓ DisgraPhi installed')"

# Check GPU (if available)
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# Run CLI tools
disgraphi-train --help
disgraphi-data --help
```

## Troubleshooting

### Common Issues

#### "No module named 'ml'"
```bash
# Make sure you installed in editable mode
pip install -e .
```

#### "CUDA out of memory"
- Reduce batch size in training config
- Use gradient accumulation
- Try 8-bit or 4-bit quantization

#### "zbar not found"
```bash
# Linux
sudo apt-get install libzbar0

# macOS
brew install zbar

# Or skip QR processing
```

#### "Slow training on CPU"
- Enable GPU in Colab or get a GPU locally
- CPU training works but is 10-50x slower

### Getting Help

If you encounter issues:

1. Check the [Troubleshooting Guide](COLAB_GUIDE.md#troubleshooting)
2. Search [existing issues](https://github.com/velocitatem/disgraPhi/issues)
3. Open a [new issue](https://github.com/velocitatem/disgraPhi/issues/new)

## Next Steps

After installation:

- **Try Colab**: [DisgraPhi_Personalization.ipynb](https://colab.research.google.com/github/velocitatem/disgraPhi/blob/main/DisgraPhi_Personalization.ipynb)
- **Read docs**: [Data Pipeline](ml/data/README.md)
- **Train locally**: `disgraphi-train --help`
- **Generate packets**: `disgraphi-data generate-packet --help`

## Updating

### Update from Git

```bash
cd disgraPhi
git pull
pip install -e . --upgrade
```

### Update from PyPI (when available)

```bash
pip install --upgrade disgraphi
```

## Uninstallation

```bash
pip uninstall disgraphi
```

To remove completely:
```bash
rm -rf disgraPhi  # Remove cloned repository
```
