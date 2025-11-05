# DisgraPhi LoRA Training - Implementation Checklist

This checklist breaks down the research plan into concrete, actionable tasks. Check off items as you complete them.

---

## 🔧 Setup & Environment (Before Phase 1)

- [ ] Install TRL and Liger Kernel
  ```bash
  pip install trl liger-kernel
  ```
- [ ] Test current setup on 12GB GPU (laptop 4080)
  ```bash
  python ml/models/train.py --model_provider smolvlm-256m --sample_ratio 0.1 --num_train_epochs 1
  ```
- [ ] Measure baseline VRAM usage
  ```bash
  nvidia-smi dmon -s u  # Monitor during training
  ```
- [ ] Document baseline metrics:
  - [ ] VRAM used: _____ GB
  - [ ] Batch size: _____
  - [ ] Training speed: _____ samples/sec
  - [ ] CER on validation: _____%

- [ ] Set up Google Colab account and test GPU access
- [ ] Verify HuggingFace authentication: `huggingface-cli whoami`
- [ ] Create project branch: `git checkout -b feature/trl-lora-optimization`

---

## 📦 Phase 1: DeepSeek-OCR Integration (Days 1-3)

### Task 1.1: Create DeepSeek-OCR Provider
- [ ] Read DeepSeek-OCR model card: https://huggingface.co/deepseek-ai/DeepSeek-OCR
- [ ] Check model architecture and tokenizer format
- [ ] Create file: `ml/models/providers/deepseek_ocr.py`
  - [ ] Implement `__init__` with LoRA config
  - [ ] Implement `forward` method
  - [ ] Implement `generate` method
  - [ ] Implement `prepare_training_batch` (with DeepSeek's input format)
  - [ ] Implement `save_adapter`
  - [ ] Implement `load_adapter`
  - [ ] Implement `set_adapter`
  - [ ] Implement `get_adapter_state_dict`
  - [ ] Implement `get_trainable_parameters`

### Task 1.2: Integrate into Factory
- [ ] Update `ml/models/providers/__init__.py`:
  ```python
  from .deepseek_ocr import DeepSeekOCR
  # Add to create_model function
  elif model_type.startswith("deepseek-ocr"):
      return DeepSeekOCR(...)
  ```

### Task 1.3: Test DeepSeek-OCR
- [ ] Test model loading:
  ```python
  from ml.models.providers import create_model
  model = create_model("deepseek-ocr", load_in_4bit=True, lora_r=8)
  print(model.get_trainable_parameters())
  ```
- [ ] Test inference on sample image
- [ ] Test forward pass with dummy batch
- [ ] Test adapter save/load

### Task 1.4: Memory Profiling
- [ ] Train on 10 samples, measure VRAM with:
  - [ ] 4-bit, r=8, batch=1: _____ GB
  - [ ] 4-bit, r=16, batch=1: _____ GB
  - [ ] 4-bit, r=16, batch=2: _____ GB
  - [ ] 4-bit, r=32, batch=1: _____ GB

- [ ] Document safe config for 12GB GPU in `RESEARCH.md`

### Task 1.5: Quick Validation Training
- [ ] Train bootstrap on 10% IAM dataset:
  ```bash
  python ml/models/train.py \
    --model_provider deepseek-ocr \
    --dataset_type iam \
    --sample_ratio 0.1 \
    --num_train_epochs 2 \
    --load_in_4bit \
    --lora_r 16 \
    --lora_alpha 32
  ```
- [ ] Check training completes without OOM
- [ ] Evaluate on validation set
- [ ] Record CER: _____%

### Task 1.6: Benchmark Against Existing Models
- [ ] Train same config on SmolVLM-256M
- [ ] Train same config on Qwen3-VL-2B (if fits in 12GB)
- [ ] Create comparison table:

| Model | VRAM (GB) | CER (%) | WER (%) | Speed (samples/s) | Adapter Size (MB) |
|-------|-----------|---------|---------|-------------------|-------------------|
| DeepSeek-OCR | | | | | |
| SmolVLM-256M | | | | | |
| Qwen3-VL-2B | | | | | |

- [ ] Add table to `ml/BENCHMARKS.md` (create file)

### Phase 1 Checkpoint
- [ ] DeepSeek-OCR trains successfully on 12GB GPU
- [ ] CER is competitive (within 20% of best existing model)
- [ ] Code reviewed and tested
- [ ] Commit: `git commit -m "feat: Add DeepSeek-OCR model provider"`

---

## 🚀 Phase 2: TRL Migration & Memory Optimization (Days 4-7)

### Task 2.1: Create TRL Training Script
- [ ] Copy `ml/models/train.py` → `ml/models/train_trl.py`
- [ ] Replace imports:
  ```python
  from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
  ```
- [ ] Update `TrainingParams` → `SFTConfig` conversion
- [ ] Replace `Trainer` with `SFTTrainer`

### Task 2.2: Implement Liger Kernel
- [ ] Add Liger Kernel initialization:
  ```python
  from liger_kernel.transformers import apply_liger_kernel_to_llama
  apply_liger_kernel_to_llama()  # or qwen/deepseek variant
  ```
- [ ] Test if it works (may need model-specific variant)
- [ ] If not available for DeepSeek: Document and skip

### Task 2.3: Implement Packing
- [ ] Enable packing in `SFTConfig`:
  ```python
  packing=True,
  dataset_text_field="text",
  max_seq_length=512,
  ```
- [ ] Update dataset to provide text field (if needed)
- [ ] Test training with packing enabled

### Task 2.4: Implement AdamW8bit
- [ ] Update optimizer in config:
  ```python
  optim="adamw_8bit"
  ```
- [ ] Verify it loads correctly
- [ ] Measure VRAM savings

### Task 2.5: Add Activation Offloading (Optional)
- [ ] Test if needed:
  ```python
  # In SFTConfig
  activation_checkpointing=True  # May already be enabled
  ```
- [ ] Benchmark: Is it worth the speed tradeoff?
- [ ] Document findings

### Task 2.6: Add Vision vs Language Layer Option
- [ ] Add parameter to `TrainingParams`:
  ```python
  finetune_vision_layers: bool = False
  finetune_language_layers: bool = True
  ```
- [ ] Implement layer freezing in providers
- [ ] Test on DeepSeek-OCR

### Task 2.7: Update LoRA Defaults
- [ ] Change defaults in `TrainingParams`:
  ```python
  lora_r: int = 16  # was 8
  lora_alpha: int = 32  # was 16
  ```
- [ ] Update all provider defaults

### Task 2.8: Memory Comparison Experiment
- [ ] Train same model with vanilla Trainer (baseline):
  - [ ] Record VRAM: _____ GB
  - [ ] Record max batch size: _____
  - [ ] Record speed: _____ samples/sec

- [ ] Train with TRL + all optimizations:
  - [ ] Record VRAM: _____ GB
  - [ ] Record max batch size: _____
  - [ ] Record speed: _____ samples/sec

- [ ] Calculate improvement:
  - [ ] VRAM reduction: _____%
  - [ ] Batch size increase: _____x
  - [ ] Speed change: _____%

### Task 2.9: Quality Validation
- [ ] Train full model on 50% IAM with TRL
- [ ] Compare CER to vanilla Trainer baseline:
  - [ ] Vanilla CER: _____%
  - [ ] TRL CER: _____%
  - [ ] Difference: _____%
- [ ] Acceptable if within ±1%

### Task 2.10: Update Documentation
- [ ] Document TRL setup in `README.md`
- [ ] Add memory optimization guide
- [ ] Update training examples

### Phase 2 Checkpoint
- [ ] TRL training works end-to-end
- [ ] ≥50% memory reduction achieved (or documented if not)
- [ ] No quality regression
- [ ] Can increase batch size on 12GB GPU
- [ ] Commit: `git commit -m "feat: Migrate to TRL with memory optimizations"`

---

## 🎛️ Phase 3: Unified Training Orchestration (Days 8-12)

### Task 3.1: Design Config Schema
- [ ] Create `ml/configs/schema.yaml` with JSON schema
- [ ] Define sections:
  - [ ] `hardware`: GPU type, VRAM, optimization level
  - [ ] `model`: Provider, quantization, LoRA config
  - [ ] `training`: Dataset, epochs, batch size, optimizer
  - [ ] `memory`: Optimization flags
  - [ ] `checkpointing`: Save strategy, HF upload
  - [ ] `logging`: TensorBoard, eval frequency

### Task 3.2: Create Hardware Profiles
- [ ] `ml/configs/hardware/laptop_12gb.yaml`:
  ```yaml
  hardware:
    device: cuda
    vram_gb: 12
    optimization: aggressive
  memory:
    use_liger_kernel: true
    use_packing: true
    gradient_checkpointing: true
  ```

- [ ] `ml/configs/hardware/colab_t4.yaml` (16GB, balanced optimization)
- [ ] `ml/configs/hardware/a100_40gb.yaml` (40GB, conservative)

### Task 3.3: Create Model Profiles
- [ ] `ml/configs/models/deepseek_ocr.yaml`
- [ ] `ml/configs/models/smolvlm_256m.yaml`
- [ ] `ml/configs/models/qwen3_vl_2b.yaml`

### Task 3.4: Create Training Profiles
- [ ] `ml/configs/training/bootstrap.yaml`
- [ ] `ml/configs/training/personalize.yaml`
- [ ] `ml/configs/training/quick_test.yaml` (10% data, 1 epoch)

### Task 3.5: Create Presets (Combined Configs)
- [ ] `ml/configs/presets/laptop_bootstrap_deepseek.yaml`
- [ ] `ml/configs/presets/colab_bootstrap_qwen3.yaml`
- [ ] `ml/configs/presets/personalize_laptop.yaml`

### Task 3.6: Implement Config Loader
- [ ] Create `ml/config.py`:
  - [ ] `load_config(path)` function
  - [ ] Config merging logic (compose multiple YAMLs)
  - [ ] Schema validation
  - [ ] Default value handling
  - [ ] CLI override support

### Task 3.7: Implement VRAM Budget Calculator
- [ ] Add to `ml/config.py`:
  ```python
  def calculate_vram_budget(
      model_size_b: float,
      quantization: str,
      lora_r: int,
      batch_size: int
  ) -> dict:
      # Returns estimated VRAM and recommendations
      pass
  ```
- [ ] Integrate into config loader
- [ ] Warn if config exceeds available VRAM
- [ ] Suggest optimizations if tight on memory

### Task 3.8: Update train.py for Config System
- [ ] Add config loading to `ml/train.py`:
  ```python
  parser.add_argument('--config', type=str)
  parser.add_argument('--hardware', type=str)
  parser.add_argument('--model', type=str)
  parser.add_argument('--training', type=str)
  ```
- [ ] Merge configs and CLI overrides
- [ ] Validate before training
- [ ] Display final config summary

### Task 3.9: Create Experiment Comparison Tool
- [ ] Create `ml/compare_experiments.py`:
  - [ ] Parse TensorBoard logs
  - [ ] Extract metrics (CER, WER, NED, loss)
  - [ ] Generate comparison table
  - [ ] Export to Markdown
  - [ ] Optional: Generate plots

- [ ] Test on existing experiments

### Task 3.10: Simplify Makefile Commands
- [ ] Add to `Makefile`:
  ```makefile
  train-bootstrap:
      python ml/train.py --config configs/presets/laptop_bootstrap_deepseek.yaml

  train-personalize:
      python ml/train.py --config configs/presets/personalize_laptop.yaml

  train-quick-test:
      python ml/train.py --config configs/presets/quick_test.yaml

  compare-experiments:
      python ml/compare_experiments.py --experiments ml/checkpoints/logs/*
  ```

### Task 3.11: Deprecate Legacy Scripts
- [ ] Move `ml/ray_train.py` → `ml/legacy/ray_train.py`
- [ ] Move `ml/train_all_bootstrap.sh` → `ml/legacy/train_all_bootstrap.sh`
- [ ] Add deprecation notice to legacy files
- [ ] Update README to reference new system

### Phase 3 Checkpoint
- [ ] Config system works end-to-end
- [ ] Can compose configs easily
- [ ] VRAM calculator prevents OOM errors
- [ ] Experiment comparison tool works
- [ ] Simple Makefile commands
- [ ] Commit: `git commit -m "feat: Add unified config-driven training system"`

---

## 📓 Phase 4: Google Colab Integration (Days 13-16)

### Task 4.1: Create Bootstrap Training Notebook
- [ ] Create `notebooks/colab_bootstrap_training.ipynb`
- [ ] Section 1: Setup
  - [ ] Install dependencies
  - [ ] Check GPU type and VRAM
  - [ ] Authenticate with HuggingFace
  - [ ] Mount Google Drive

- [ ] Section 2: Dataset Preparation
  - [ ] Download IAM dataset (with caching to Drive)
  - [ ] Show sample images
  - [ ] Display dataset statistics

- [ ] Section 3: Training Configuration
  - [ ] Model selection dropdown (DeepSeek-OCR, SmolVLM, Qwen3)
  - [ ] Training parameters (epochs, batch size, LoRA rank)
  - [ ] Estimated training time

- [ ] Section 4: Training
  - [ ] Load config (colab_t4.yaml)
  - [ ] Start training with progress bar
  - [ ] Display live metrics (loss, CER)
  - [ ] Save checkpoints to Drive every 500 steps

- [ ] Section 5: Evaluation & Upload
  - [ ] Evaluate on test set
  - [ ] Show example predictions
  - [ ] Upload adapter to HuggingFace
  - [ ] Provide download link

### Task 4.2: Create Personalization Notebook
- [ ] Create `notebooks/colab_personalization.ipynb`
- [ ] Section 1: Setup (same as bootstrap)

- [ ] Section 2: Upload Handwriting Samples
  - [ ] File upload widget
  - [ ] Automatic manifest.json generation
  - [ ] Show uploaded images with labels
  - [ ] Validate dataset (check quality, count)

- [ ] Section 3: Download Bootstrap Adapter
  - [ ] Input: HuggingFace repo ID
  - [ ] Download adapter to Drive
  - [ ] Verify adapter loads

- [ ] Section 4: Personalization Training
  - [ ] Train with augmentation
  - [ ] Quick training (15-30 min)
  - [ ] Live progress updates

- [ ] Section 5: Testing & Download
  - [ ] Test on uploaded samples
  - [ ] Show before/after predictions
  - [ ] Download personalized adapter
  - [ ] Optional: Upload to HuggingFace

### Task 4.3: Add Drive Checkpointing
- [ ] Create `ml/utils/drive_checkpoint.py`:
  - [ ] Save checkpoint to Drive path
  - [ ] Resume from Drive checkpoint
  - [ ] Auto-detect disconnection
  - [ ] Clean up old checkpoints

- [ ] Integrate into training loop
- [ ] Test disconnection → resume flow

### Task 4.4: Optimize for Colab Free Tier
- [ ] Test on actual free tier (T4, not V100/A100)
- [ ] Ensure bootstrap completes in <6 hours
- [ ] Ensure personalization completes in <30 min
- [ ] Add progress estimates based on T4 benchmarks

### Task 4.5: Error Handling & Recovery
- [ ] Add error messages for common issues:
  - [ ] No GPU available
  - [ ] Out of memory
  - [ ] Disconnection during training
  - [ ] HuggingFace auth failure
  - [ ] Drive mount issues

- [ ] Add recovery instructions
- [ ] Add automatic retries where appropriate

### Task 4.6: Create Troubleshooting Guide
- [ ] Create `docs/COLAB_TROUBLESHOOTING.md`:
  - [ ] GPU not available → wait and retry
  - [ ] OOM → reduce batch size
  - [ ] Slow training → check GPU type
  - [ ] Lost connection → resume from checkpoint
  - [ ] Upload failed → retry with HF token

### Task 4.7: Test End-to-End on Colab
- [ ] Run bootstrap notebook on free tier
- [ ] Time the training: _____ hours
- [ ] Record final CER: _____%
- [ ] Run personalization notebook
- [ ] Time the training: _____ minutes
- [ ] Test disconnection recovery
- [ ] Get external user to test (if possible)

### Task 4.8: Create Video Tutorial (Optional)
- [ ] Record screen capture of full workflow
- [ ] Voiceover explaining each step
- [ ] Upload to YouTube
- [ ] Link from README

### Phase 4 Checkpoint
- [ ] Both notebooks work on Colab free tier
- [ ] Bootstrap completes in <6 hours
- [ ] Personalization completes in <30 min
- [ ] Clear error messages and recovery
- [ ] Non-technical user can follow
- [ ] Commit: `git commit -m "feat: Add Google Colab notebooks with Drive checkpointing"`

---

## 📊 Phase 5: Dataset Handling & Augmentation (Days 17-19)

### Task 5.1: Add Sample Limit to ManifestDataset
- [ ] Update `ml/data/datasets.py`:
  ```python
  class ManifestDataset:
      def __init__(self, ..., max_samples=None):
          # Limit dataset size for personalization
  ```
- [ ] Add validation warning if >200 samples for personalization
- [ ] Document recommended dataset sizes in docstring

### Task 5.2: Implement Handwriting Augmentations
- [ ] Add to `ml/data/augmentation.py`:
  - [ ] Deskew/rotation (±2°)
  - [ ] Mild Gaussian blur (sigma=0.5-1.0)
  - [ ] JPEG compression (quality=80-95)
  - [ ] Verify existing augmentations still work:
    - [x] Elastic distortion
    - [x] Slant variation
    - [x] Baseline drift

### Task 5.3: Create Augmentation Presets
- [ ] Add preset system:
  ```python
  AUGMENTATION_PRESETS = {
      'conservative': {'rotation': 1, 'blur': 0.3, 'strength': 0.5},
      'standard': {'rotation': 2, 'blur': 0.7, 'strength': 0.7},
      'aggressive': {'rotation': 3, 'blur': 1.0, 'strength': 0.9},
  }
  ```
- [ ] Integrate with config system
- [ ] Update CLI to accept preset name

### Task 5.4: Create Augmentation Visualization Tool
- [ ] Create `ml/data/visualize_augmentation.py`:
  - [ ] Load sample images
  - [ ] Apply augmentations with different strengths
  - [ ] Display before/after grid
  - [ ] Save comparison images

- [ ] Test on IAM samples
- [ ] Add usage examples to docs

### Task 5.5: Dataset Collection Best Practices Guide
- [ ] Create `docs/DATA_COLLECTION_GUIDE.md`:
  - [ ] How many samples needed (50-200)
  - [ ] What to write (diverse sentences, all characters)
  - [ ] How to photograph (lighting, angle, resolution)
  - [ ] Common mistakes to avoid
  - [ ] Quality checklist

### Task 5.6: Create Practice Packet Template
- [ ] Create `ml/data/templates/practice_packet.html` or PDF:
  - [ ] Printable template with lines
  - [ ] Example sentences to write
  - [ ] QR code for identification
  - [ ] Instructions for users

- [ ] Test: Print → write → photograph → process

### Task 5.7: Create Data Validation Script
- [ ] Create `ml/data/validate_dataset.py`:
  - [ ] Check image resolution (minimum 300x100)
  - [ ] Check contrast (detect too bright/dark)
  - [ ] Check text labels (no empty strings)
  - [ ] Detect potential OCR errors in labels
  - [ ] Generate quality report

- [ ] Integrate with personalization workflow

### Task 5.8: Experimental Validation of Augmentation
- [ ] Train model without augmentation (baseline)
- [ ] Train model with conservative augmentation
- [ ] Train model with standard augmentation
- [ ] Train model with aggressive augmentation
- [ ] Compare CER on held-out test set:

| Augmentation | CER (%) | Overfitting? |
|--------------|---------|--------------|
| None | | |
| Conservative | | |
| Standard | | |
| Aggressive | | |

- [ ] Document findings in `RESEARCH.md`

### Phase 5 Checkpoint
- [ ] Dataset limits implemented and validated
- [ ] New augmentations work correctly
- [ ] Visualization tool helps users tune augmentation
- [ ] Data collection guide is clear
- [ ] Practice packet template is usable
- [ ] Commit: `git commit -m "feat: Add dataset handling improvements and augmentation presets"`

---

## 🏆 Phase 6: Benchmarking Framework (Days 20-23)

### Task 6.1: Create Benchmark Script
- [ ] Create `ml/benchmark.py`:
  ```python
  def benchmark_model(
      model_path: str,
      test_dataset: str,
      output_file: str
  ) -> dict:
      # Returns comprehensive metrics
      pass
  ```

### Task 6.2: Implement Metrics Collection
- [ ] Metrics to collect:
  - [ ] CER, WER, NED (already implemented)
  - [ ] Accuracy (exact match rate)
  - [ ] VRAM usage during inference
  - [ ] Latency (ms per image)
  - [ ] Throughput (images/sec)
  - [ ] Adapter size (MB)
  - [ ] Total parameters / trainable parameters

### Task 6.3: Add Per-Writer Evaluation
- [ ] IAM dataset has writer IDs
- [ ] Compute metrics per writer:
  - [ ] Mean CER across writers
  - [ ] Std dev (shows generalization)
  - [ ] Worst writer (identifies failure modes)
  - [ ] Best writer

- [ ] Generate per-writer breakdown table

### Task 6.4: Create Benchmark Suite
- [ ] Define standard benchmarks:
  - [ ] IAM test set (standard benchmark)
  - [ ] User-collected test set (if available)
  - [ ] Synthetic difficult examples (cursive, messy)

- [ ] Run all models on all benchmarks
- [ ] Store results in JSON format

### Task 6.5: Create Leaderboard
- [ ] Create `ml/LEADERBOARD.md`:
  - [ ] Bootstrap models table (IAM only)
  - [ ] Personalized models table (IAM + user)
  - [ ] Sortable by CER, speed, VRAM
  - [ ] Links to HuggingFace repos

- [ ] Auto-generate from benchmark results

### Task 6.6: Inference Benchmarking
- [ ] Create `ml/benchmark_inference.py`:
  - [ ] Load model once, run N inferences
  - [ ] Measure latency (p50, p95, p99)
  - [ ] Measure throughput with batching
  - [ ] Measure VRAM during inference

- [ ] Compare models:

| Model | Latency (ms) | Throughput (img/s) | VRAM (GB) |
|-------|--------------|-------------------|-----------|
| DeepSeek-OCR | | | |
| SmolVLM-256M | | | |
| Qwen3-VL-2B | | | |

### Task 6.7: Error Analysis Tool
- [ ] Create `ml/analyze_errors.py`:
  - [ ] Show worst predictions (highest CER)
  - [ ] Character confusion matrix (which chars get confused)
  - [ ] Error type breakdown (insertions vs deletions vs substitutions)
  - [ ] Visualize errors on images

### Task 6.8: Optional: Streamlit Dashboard
- [ ] Create `ml/dashboard.py`:
  - [ ] Interactive benchmark viewer
  - [ ] Side-by-side model predictions
  - [ ] Upload image → get predictions from all models
  - [ ] Error analysis visualization

- [ ] Run with: `streamlit run ml/dashboard.py`

### Task 6.9: Run Comprehensive Benchmarks
- [ ] Benchmark all models on IAM test set
- [ ] Record results in `ml/benchmarks/results.json`
- [ ] Generate leaderboard
- [ ] Update `LEADERBOARD.md`
- [ ] Create summary plots (CER vs VRAM, CER vs speed)

### Phase 6 Checkpoint
- [ ] Benchmarking framework is comprehensive
- [ ] Leaderboard shows clear winners for each use case
- [ ] Results are reproducible
- [ ] Error analysis provides insights
- [ ] Commit: `git commit -m "feat: Add comprehensive benchmarking framework"`

---

## 📚 Phase 7: Documentation & User Guide (Days 24-26)

### Task 7.1: Update README.md
- [ ] Add project overview
- [ ] Add quick start guide (3 commands to first training)
- [ ] Add features list
- [ ] Add model comparison table
- [ ] Add links to notebooks, docs
- [ ] Add badges (build status, license, etc.)
- [ ] Add example images (before/after OCR)

### Task 7.2: Create Training Guide
- [ ] Create `docs/TRAINING_GUIDE.md`:
  - [ ] Step-by-step bootstrap training
  - [ ] Step-by-step personalization
  - [ ] Config system explained
  - [ ] Hardware requirements
  - [ ] Expected training times
  - [ ] Tips for improving quality

### Task 7.3: Create Troubleshooting Guide
- [ ] Create `docs/TROUBLESHOOTING.md`:
  - [ ] CUDA out of memory → reduce batch size, use 4-bit
  - [ ] Slow training → check GPU utilization, use faster GPU
  - [ ] Poor quality → more data, better augmentation, larger model
  - [ ] Adapter not loading → version mismatch, path issues
  - [ ] Import errors → dependency versions

- [ ] Add solutions for each error

### Task 7.4: Create Model Selection Guide
- [ ] Create `docs/MODEL_SELECTION.md`:
  - [ ] Decision tree: VRAM → model choice
  - [ ] Speed vs accuracy tradeoffs
  - [ ] When to use DeepSeek-OCR vs SmolVLM vs Qwen3
  - [ ] Quantization guide (4-bit vs 8-bit vs full)

### Task 7.5: Create Example Use Cases
- [ ] Add to `docs/USE_CASES.md`:
  - [ ] Student digitizing handwritten notes
  - [ ] Doctor transcribing prescriptions
  - [ ] Historian digitizing archives
  - [ ] Artist converting sketches
  - [ ] Language learner practicing writing

- [ ] Include expected results for each

### Task 7.6: Create FAQ
- [ ] Create `docs/FAQ.md`:
  - [ ] How much data do I need? (50-200 samples)
  - [ ] How long does training take? (depends on GPU)
  - [ ] Can I use this offline? (yes, after download)
  - [ ] What languages are supported? (any with Unicode)
  - [ ] Can I fine-tune for printed text? (yes, same process)
  - [ ] How do I improve accuracy? (more data, augmentation, larger model)

### Task 7.7: Create Contribution Guide
- [ ] Create `CONTRIBUTING.md`:
  - [ ] How to add new model provider
  - [ ] Code style guide (follows CLAUDE.md tenets)
  - [ ] Testing requirements
  - [ ] PR process
  - [ ] How to report bugs
  - [ ] How to request features

### Task 7.8: Create API Documentation
- [ ] Document all public APIs:
  - [ ] `create_model()` function
  - [ ] `BaseVisionLanguageModel` interface
  - [ ] Config system API
  - [ ] Benchmark functions

- [ ] Add docstrings to all public functions
- [ ] Generate API docs with Sphinx (optional)

### Task 7.9: Create Deployment Guide
- [ ] Create `docs/DEPLOYMENT.md`:
  - [ ] How to use adapter for inference
  - [ ] FastAPI inference server (already exists in ml/inference.py)
  - [ ] Batch processing scripts
  - [ ] Cloud deployment (AWS, GCP, Azure)
  - [ ] Edge deployment (Raspberry Pi, mobile)

### Task 7.10: Final Polish
- [ ] Proofread all documentation
- [ ] Check all links work
- [ ] Ensure code examples run
- [ ] Add table of contents to long docs
- [ ] Add diagrams where helpful
- [ ] Spellcheck

### Phase 7 Checkpoint
- [ ] Complete, clear documentation
- [ ] Non-technical users can get started
- [ ] All common issues documented
- [ ] Contribution guide encourages community involvement
- [ ] Commit: `git commit -m "docs: Add comprehensive user documentation"`

---

## 🎯 Final Integration & Testing (Days 27+)

### Task 8.1: End-to-End Testing
- [ ] Fresh environment test (new Docker container):
  - [ ] Clone repo
  - [ ] Follow README to first training
  - [ ] Verify everything works
  - [ ] Time the process: _____ minutes

### Task 8.2: External User Testing
- [ ] Get 2-3 external users to test:
  - [ ] Follow Colab notebook
  - [ ] Provide feedback
  - [ ] Report any issues

- [ ] Fix all reported issues

### Task 8.3: Performance Validation
- [ ] Verify success metrics:
  - [x] Memory efficiency ≥50% improvement: _____%
  - [ ] Training speed: no major regression
  - [ ] Model quality: CER ≤5% on IAM
  - [ ] Colab training: <6 hours
  - [ ] Personalization: <30 minutes

### Task 8.4: Code Review & Cleanup
- [ ] Review all code against CLAUDE.md tenets:
  - [ ] No bloated code
  - [ ] Minimal comments (code should be self-explanatory)
  - [ ] No redundant variables
  - [ ] Efficient list comprehensions
  - [ ] Proper error handling

- [ ] Remove any debug code
- [ ] Clean up TODOs

### Task 8.5: Create Release
- [ ] Update version number
- [ ] Create CHANGELOG.md
- [ ] Tag release: `git tag v1.0.0`
- [ ] Create GitHub release with notes
- [ ] Upload all bootstrap adapters to HuggingFace
- [ ] Announce on social media / forums (optional)

### Task 8.6: Future Work Backlog
- [ ] Create GitHub issues for future features:
  - [ ] Multi-adapter composition
  - [ ] Synthetic data generation
  - [ ] Few-shot learning experiments
  - [ ] Multilingual support
  - [ ] Continuous learning

---

## 📊 Progress Tracking

**Overall Progress:** [ ] 0% → [ ] 100%

**Phase Completion:**
- [ ] Phase 0: Setup (0%)
- [ ] Phase 1: DeepSeek-OCR (0%)
- [ ] Phase 2: TRL Migration (0%)
- [ ] Phase 3: Config System (0%)
- [ ] Phase 4: Colab (0%)
- [ ] Phase 5: Datasets (0%)
- [ ] Phase 6: Benchmarking (0%)
- [ ] Phase 7: Documentation (0%)
- [ ] Phase 8: Integration (0%)

**Blockers & Issues:**
(Add any blockers here as you encounter them)

---

## 🎓 Learning Resources

As you work through this checklist, refer to:

- **TRL Docs:** https://huggingface.co/docs/trl
- **PEFT Docs:** https://huggingface.co/docs/peft
- **DeepSeek-OCR:** https://huggingface.co/deepseek-ai/DeepSeek-OCR
- **Liger Kernel:** https://github.com/linkedin/Liger-Kernel
- **CLAUDE.md:** Your project's code quality tenets

---

## ✅ Definition of Done

This project is complete when:

1. ✅ DeepSeek-OCR trains successfully on 12GB GPU
2. ✅ TRL achieves ≥50% memory reduction (or documented if not)
3. ✅ Config system replaces all legacy scripts
4. ✅ Colab notebooks work on free tier
5. ✅ Bootstrap training <6 hours on T4
6. ✅ Personalization <30 min with 100 samples
7. ✅ Comprehensive benchmarks in LEADERBOARD.md
8. ✅ Complete documentation for non-technical users
9. ✅ External user successfully trains model
10. ✅ All code follows CLAUDE.md tenets

**Let's build it! 🚀**
