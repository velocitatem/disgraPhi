# 🚀 Quick Start Guide

Get started with DisgraPhi in under 5 minutes!

## Choose Your Path

### 🌐 Option 1: Google Colab (Easiest - No Installation!)

Perfect for first-time users and quick experiments.

1. **Click the badge**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/velocitatem/disgraPhi/blob/main/DisgraPhi_Personalization.ipynb)

2. **Enable GPU**:
   - Go to `Runtime` → `Change runtime type`
   - Set Hardware accelerator to `GPU` (T4)
   - Click `Save`

3. **Follow the notebook**:
   - Run cells in order (Shift+Enter)
   - Upload 10-30 handwriting samples
   - Label what each sample says
   - Wait ~20 minutes for training
   - Download your personalized model!

**Total time: ~45 minutes** (mostly hands-off)

See [COLAB_GUIDE.md](COLAB_GUIDE.md) for detailed instructions.

---

### 💻 Option 2: Local Installation

Perfect for developers and production use.

```bash
# Clone repository
git clone https://github.com/velocitatem/disgraPhi.git
cd disgraPhi

# Create virtual environment and install
make venv
source .venv/bin/activate
```

**That's it!** Now you can use the CLI or Python API.

See [INSTALL.md](INSTALL.md) for platform-specific instructions.

---

## Quick Examples

### Train on Your Handwriting

#### Using Colab
Just click the badge above and follow the interactive notebook!

#### Using CLI
```bash
# Generate practice packet
disgraphi-data generate-packet --user-id alice --output packet.pdf

# (Print, fill out, and photograph)

# Process photos
disgraphi-data process-packet \
    --user-id alice \
    --images page1.jpg page2.jpg \
    --ground-truth packet_ground_truth.json

# Train personalized model
disgraphi-train \
    --dataset_type manifest \
    --manifest_data_dir ml/data/users/alice \
    --num_train_epochs 5 \
    --output_dir ./models
```

### Use Python API

```python
from ml.models.providers import create_model
from PIL import Image

# Load model with your adapter
model = create_model("smolvlm-256m")
model.load_adapter("path/to/your/adapter")

# Transcribe handwriting
image = Image.open("handwriting.jpg")
text = model.generate(
    pixel_values=image,
    prompt="Transcribe this handwritten text.",
    max_new_tokens=128
)
print(text)
```

### Deploy Inference API

```bash
# Set adapter path
export ML_ADAPTER_PATH="path/to/your/adapter"
export ML_MODEL_NAME="smolvlm-256m"

# Start server
python ml/inference.py
```

API available at `http://localhost:8000/docs`

---

## What You Need

### For Colab (Recommended)
- ✅ Google account (free)
- ✅ 10-30 handwriting samples (photos)
- ✅ 45 minutes of time

### For Local Setup
- ✅ Python 3.10+ 
- ✅ NVIDIA GPU (8GB+ VRAM recommended)
- ✅ 10GB free disk space

---

## Next Steps

After getting started:

1. **Read the docs**: 
   - [Colab Guide](COLAB_GUIDE.md) - Detailed Colab instructions
   - [Installation](INSTALL.md) - Local setup
   - [Data Pipeline](ml/data/README.md) - Data preparation

2. **Try advanced features**:
   - Use packet generator for structured data
   - Experiment with different models (qwen3-vl-2b, florence2-base)
   - Deploy as production API
   - Monitor training with TensorBoard

3. **Get help**:
   - [FAQ](COLAB_GUIDE.md#troubleshooting)
   - [GitHub Issues](https://github.com/velocitatem/disgraPhi/issues)
   - [Discussions](https://github.com/velocitatem/disgraPhi/discussions)

---

## Common Questions

**Q: How accurate will it be?**
A: With 20-30 samples, expect 85-95% accuracy. With 100+ samples, 95-98%.

**Q: Will it work for messy handwriting?**
A: Yes! That's exactly what DisgraPhi is designed for. More samples = better results.

**Q: Do I need coding experience?**
A: No! The Colab notebook is fully interactive - just click buttons.

**Q: Can I use it commercially?**
A: Yes! MIT license - use it however you want.

**Q: How long does training take?**
A: 15-30 minutes on Colab's free GPU, depending on sample count.

---

**Ready to start?** Pick your path above and get going! 🚀
