![](./banner.png)

# _Disgraϕ_

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/velocitatem/disgraPhi/blob/main/DisgraPhi_Personalization.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Handwriting recognition has come a long way, but it still struggles with the writing that matters most. People with dysgraphia and other handwriting difficulties often find that even the best OCR systems can't reliably read their work. This isn't surprising—these systems are trained on neat, consistent handwriting, not the varied and unconventional styles that real people actually write in.

DisgraPhi takes a different approach. Instead of trying to build one model that works for everyone, it learns your handwriting specifically. You provide a small sample of your writing, and the system fine-tunes itself to understand your unique style. The result is a personalized OCR model that actually works with how you write, not against it.

The system builds on modern vision-language models but keeps things practical. It's designed to run on consumer hardware and doesn't require massive datasets or expensive infrastructure. A few pages of your handwriting are enough to train a model that understands your writing better than any generic solution could.

This matters because between 5-20% of people struggle with handwriting in some way, and that number keeps growing. Whether it's dysgraphia, motor difficulties, or just unconventional penmanship, everyone deserves tools that work with their writing. DisgraPhi is a step toward making that happen.

## 🚀 Quick Start

### Try it in Google Colab (Easiest!)

The fastest way to get started - no installation required:

1. Click the "Open in Colab" badge above
2. Upload 10-30 images of your handwriting
3. Label what each image says
4. Train your personalized model (15-30 minutes)
5. Download and use your model

**Perfect for**: First-time users, quick experiments, no-code experience

See the [Colab Guide](COLAB_GUIDE.md) for detailed instructions.

### Install Locally

For development or production use:

```bash
# Clone the repository
git clone https://github.com/velocitatem/disgraPhi.git
cd disgraPhi

# Create virtual environment
make venv
source .venv/bin/activate

# Install package
pip install -e .
```

**Perfect for**: Developers, production deployment, batch processing

## 📦 Features

- **🎯 Personalized OCR**: Train models on your unique handwriting style
- **🔧 Easy to Use**: One-click Colab notebook or simple CLI
- **⚡ Efficient**: Runs on free Google Colab or consumer GPUs
- **🧠 State-of-the-art**: Built on modern vision-language models (SmolVLM, Qwen3-VL, Florence2)
- **📊 Few-shot Learning**: Get good results with just 10-30 samples
- **🎨 Data Augmentation**: Smart augmentation for better generalization
- **📈 Training Monitoring**: TensorBoard integration for tracking progress
- **🌐 API Ready**: Deploy as REST API for production use

