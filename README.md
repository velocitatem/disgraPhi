![](./banner.png)

# _Disgraϕ_

Handwriting recognition has come a long way, but it still struggles with the writing that matters most. People with dysgraphia and other handwriting difficulties often find that even the best OCR systems can't reliably read their work. This isn't surprising—these systems are trained on neat, consistent handwriting, not the varied and unconventional styles that real people actually write in.

DisgraPhi takes a different approach. Instead of trying to build one model that works for everyone, it learns your handwriting specifically. You provide a small sample of your writing, and the system fine-tunes itself to understand your unique style. The result is a personalized OCR model that actually works with how you write, not against it.

The system builds on modern vision-language models but keeps things practical. It's designed to run on consumer hardware and doesn't require massive datasets or expensive infrastructure. A few pages of your handwriting are enough to train a model that understands your writing better than any generic solution could.

This matters because between 5-20% of people struggle with handwriting in some way, and that number keeps growing. Whether it's dysgraphia, motor difficulties, or just unconventional penmanship, everyone deserves tools that work with their writing. DisgraPhi is a step toward making that happen.


## Features

### 🎯 Core Capabilities
- **Personalized OCR**: Train models on your handwriting in 15-30 minutes
- **Multiple Models**: DeepSeek-OCR (3B), SmolVLM (256M-2.2B), Qwen3-VL (2B-7B)
- **Memory Efficient**: Train on 12GB laptop GPUs with QLoRA + TRL optimizations
- **60% Memory Reduction**: Liger Kernel integration for efficient training
- **Google Colab Support**: Train on free tier T4 GPUs

### 📊 Training Modes
1. **Bootstrap**: Train general handwriting adapter on IAM dataset (13k samples)
2. **Personalization**: Fine-tune on your handwriting (50-200 samples recommended)

### ⚙️ Optimizations
- LoRA (Low-Rank Adaptation): r=16, alpha=32 for handwriting
- 4-bit/8-bit quantization via BitsAndBytes
- AdamW8bit optimizer (25-30% VRAM reduction)
- Gradient checkpointing
- TRL SFTTrainer with packing support
- Handwriting-specific augmentations

### 📈 Benchmarking
- Comprehensive evaluation framework
- OCR metrics: CER, WER, NED, accuracy
- Inference benchmarks: latency, throughput, VRAM
- Per-writer performance analysis
- Automatic leaderboard generation

## Quick Start

### Option 1: Google Colab (Recommended for beginners)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/colab_bootstrap_training.ipynb)

1. Open the bootstrap training notebook
2. Follow the steps (5-10 min setup, 3-5 hours training)
3. Your trained adapter will be saved to Google Drive

### Option 2: Local Training (12GB+ GPU)

```bash
# Clone repository
git clone https://github.com/velocitatem/disgraPhi.git
cd disgraPhi

# Install dependencies
pip install -r ml/requirements.txt

# Quick test (10% data, 1 epoch)
python ml/models/train_trl.py \
  --model_provider deepseek-ocr \
  --dataset_type iam \
  --sample_ratio 0.1 \
  --num_train_epochs 1

# Full bootstrap training (3 epochs, ~3-4 hours on 4080)
python ml/models/train_trl.py \
  --model_provider deepseek-ocr \
  --dataset_type iam \
  --num_train_epochs 3
```

### Option 3: Using Config Presets

```bash
# Laptop training (12GB GPU)
python ml/models/train_trl.py \
  --config ml/configs/presets/laptop_bootstrap_deepseek.yaml

# Personalization
python ml/models/train_trl.py \
  --config ml/configs/presets/laptop_personalize.yaml \
  --manifest_data_dir ./my_handwriting \
  --bootstrap_adapter_path ./checkpoints/bootstrap/final_adapter
```

## Installation

### Requirements
- Python 3.9+
- CUDA 11.8+ (for GPU training)
- 12GB+ VRAM (or Google Colab T4)

### Dependencies
```bash
pip install -r ml/requirements.txt
```

Key packages:
- `torch` - Deep learning framework
- `transformers` - HuggingFace models
- `trl>=0.9.0` - Transformer Reinforcement Learning
- `peft` - Parameter-Efficient Fine-Tuning (LoRA)
- `liger-kernel` - Memory optimization (60% reduction)
- `bitsandbytes` - Quantization

## Model Zoo

| Model | Size | VRAM (4-bit) | Speed | Best For |
|-------|------|--------------|-------|----------|
| **DeepSeek-OCR** | 3B | ~3GB | Medium | Handwriting, OCR |
| **SmolVLM-256M** | 256M | ~0.5GB | Fast | Edge devices, prototyping |
| **SmolVLM-500M** | 500M | ~1GB | Fast | Balanced efficiency |
| **Qwen3-VL-2B** | 2B | ~2GB | Medium | General vision-language |
| **Qwen3-VL-4B** | 4B | ~3.5GB | Slower | High accuracy |

*Recommended: DeepSeek-OCR for handwriting recognition*

## Training Guide

### 1. Bootstrap Training (IAM Dataset)

Trains a general handwriting adapter that works for most people:

```bash
python ml/models/train_trl.py \
  --model_provider deepseek-ocr \
  --dataset_type iam \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --load_in_4bit
```

**Expected Results:**
- Training time: 3-5 hours (T4), 2-3 hours (4080)
- CER: <5% on IAM test set
- Adapter size: ~50-200MB

### 2. Personalization (Your Handwriting)

Fine-tune on your handwriting samples:

**Collect Data:**
1. Write 50-200 sentences (diverse text)
2. Photograph each line clearly
3. Create `manifest.json`:

```json
{
  "train": [
    {"image": "line001.jpg", "text": "The quick brown fox..."},
    {"image": "line002.jpg", "text": "Pack my box with..."}
  ],
  "val": [...],
  "test": [...]
}
```

**Train:**
```bash
python ml/models/train_trl.py \
  --model_provider deepseek-ocr \
  --dataset_type manifest \
  --manifest_data_dir ./my_handwriting \
  --bootstrap_adapter_path ./checkpoints/bootstrap/final_adapter \
  --num_train_epochs 2 \
  --learning_rate 3e-4 \
  --augment \
  --augment_strength 0.7
```

**Expected Results:**
- Training time: 15-30 minutes
- CER: <3% on your handwriting
- Significant improvement over bootstrap

## Configuration System

DisgraPhi uses a flexible YAML-based config system:

```yaml
# Example: ml/configs/presets/laptop_bootstrap_deepseek.yaml
hardware:
  vram_gb: 12
  optimization: aggressive

model:
  provider: deepseek-ocr
  quantization: 4bit
  lora:
    r: 16
    alpha: 32

training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  learning_rate: 2e-4

memory:
  use_liger_kernel: true
  optim: adamw_8bit
```

**Available Configs:**
- Hardware: `laptop_12gb`, `colab_t4`, `a100_40gb`
- Models: `deepseek_ocr`, `smolvlm_256m`, `qwen3_vl_2b`
- Training: `bootstrap`, `personalize`, `quick_test`

## Benchmarking

Compare models systematically:

```bash
# Benchmark multiple models
python ml/benchmark.py \
  --models deepseek-ocr smolvlm-256m qwen3-vl-2b \
  --quantization 4bit 4bit 4bit \
  --output ml/LEADERBOARD.md

# Benchmark with adapters
python ml/benchmark.py \
  --models deepseek-ocr \
  --adapters ./checkpoints/bootstrap/final_adapter \
  --max_samples 100  # Quick test
```

**Metrics:**
- OCR: CER, WER, NED, accuracy
- Performance: Latency (p50, p95, p99), throughput
- Resources: VRAM usage, adapter size
- Per-writer analysis

## Project Structure

```
disgraPhi/
├── ml/
│   ├── models/
│   │   ├── providers/          # Model implementations
│   │   │   ├── deepseek_ocr.py # DeepSeek-OCR
│   │   │   ├── smolvlm.py      # SmolVLM
│   │   │   ├── qwen3.py        # Qwen3-VL
│   │   │   └── base.py         # Base interface
│   │   ├── train.py            # Original trainer
│   │   ├── train_trl.py        # TRL-optimized trainer
│   │   └── eval.py             # OCR metrics
│   ├── data/
│   │   ├── datasets.py         # IAMDataset, ManifestDataset
│   │   ├── augmentation.py     # Handwriting augmentation
│   │   └── etl.py              # Data processing
│   ├── configs/                # YAML configs
│   │   ├── hardware/           # Hardware profiles
│   │   ├── models/             # Model configs
│   │   ├── training/           # Training configs
│   │   └── presets/            # Combined presets
│   ├── config.py               # Config loader + VRAM calculator
│   ├── benchmark.py            # Benchmarking framework
│   └── compare_experiments.py  # Experiment comparison
├── notebooks/
│   ├── colab_bootstrap_training.ipynb
│   └── colab_personalization.ipynb
└── docs/
    ├── RESEARCH.md             # Research findings
    └── IMPLEMENTATION_CHECKLIST.md
```

## Memory Optimization

DisgraPhi implements aggressive memory optimizations for 12GB GPUs:

| Technique | Memory Reduction | Implementation |
|-----------|------------------|----------------|
| **Liger Kernel** | 60% | Triton kernels for LLM training |
| **QLoRA (4-bit)** | 75% | NF4 quantization |
| **AdamW8bit** | 25-30% | 8-bit optimizer states |
| **Gradient Checkpointing** | 40-60% | Trade compute for memory |
| **LoRA** | 99%+ | Train 0.1-0.5% of parameters |

**VRAM Budget Example (DeepSeek-OCR 3B):**
```
Model (4-bit):           ~2.4 GB
LoRA adapters (r=16):    ~0.8 GB
Optimizer (8-bit):       ~1.5 GB
Activations (batch=2):   ~2.5 GB (with checkpointing)
Gradients:               ~1.8 GB
Buffer:                  ~1.0 GB
Total:                   ~10.0 GB (fits in 12GB)

With Liger Kernel:       ~4.0 GB (massive headroom!)
```

## Advanced Features

### Augmentation Presets

```python
from ml.data.augmentation import get_augmentation

# For small datasets (<50 samples)
aug = get_augmentation('aggressive')

# Balanced (50-100 samples)
aug = get_augmentation('standard')

# Large datasets (>100 samples)
aug = get_augmentation('conservative')

# Apply to image
augmented_image = aug(image)
```

### Custom Model Integration

Add new models by implementing `BaseVisionLanguageModel`:

```python
from ml.models.providers.base import BaseVisionLanguageModel

class MyModel(BaseVisionLanguageModel):
    def forward(self, ...): ...
    def generate(self, ...): ...
    def prepare_training_batch(self, ...): ...
    # ... other required methods

# Register in ml/models/providers/__init__.py
MODEL_REGISTRY['my-model'] = MyModel
```

### Experiment Comparison

```bash
# Compare all experiments in logs directory
python ml/compare_experiments.py \
  --logs_dir ml/checkpoints/logs \
  --output ml/EXPERIMENT_COMPARISON.md

# Generate JSON report
python ml/compare_experiments.py \
  --logs_dir ml/checkpoints/logs \
  --output ml/comparison.md \
  --json ml/results.json
```

## Troubleshooting

### Out of Memory (OOM)

1. **Reduce batch size:**
   ```bash
   --per_device_train_batch_size 1
   ```

2. **Enable all optimizations:**
   ```bash
   --load_in_4bit --use_liger_kernel --optim adamw_8bit
   ```

3. **Use smaller model:**
   ```bash
   --model_provider smolvlm-256m
   ```

4. **Enable activation offloading (slower but saves memory):**
   ```bash
   --activation_offloading
   ```

### Slow Training

- Use larger batch size if VRAM allows
- Disable gradient checkpointing on high-VRAM GPUs
- Use full precision (bf16/fp16) instead of 4-bit
- Try faster model (SmolVLM-256M)

### Poor Accuracy

- Collect more diverse training data (50-200 samples)
- Enable augmentation: `--augment --augment_strength 0.7`
- Train for more epochs: `--num_train_epochs 5`
- Use larger model (Qwen3-VL-4B)
- Check data quality (clear images, correct labels)

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Follow the code style (see CLAUDE.md)
4. Add tests if applicable
5. Submit a pull request

**Areas for contribution:**
- New model providers
- Additional augmentation techniques
- Mobile/edge deployment
- Multi-language support
- Better data collection tools

## Citation

```bibtex
@software{disgraphi2025,
  author = {Velocitatem},
  title = {DisgraPhi: Personalized Handwriting Recognition},
  year = {2025},
  url = {https://github.com/velocitatem/disgraPhi}
}
```

## License

Apache 2.0 - See LICENSE file

## Acknowledgments

- IAM Handwriting Database
- HuggingFace Transformers & TRL
- DeepSeek-OCR, SmolVLM, Qwen3-VL teams
- PEFT (LoRA) library
- Liger Kernel for memory optimization

## Support

- **Issues**: [GitHub Issues](https://github.com/velocitatem/disgraPhi/issues)
- **Documentation**: [Research & Implementation Guide](RESEARCH.md)
- **Discord**: Coming soon

---

**DisgraPhi** - Making handwriting recognition work for everyone, especially those who need it most.
