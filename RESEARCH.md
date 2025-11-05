# DisgraPhi LoRA Training Research & Implementation Plan

## Executive Summary

This document outlines a comprehensive plan to transform DisgraPhi into an efficient, accessible handwriting recognition system using LoRA adapters. Target: anyone can train personalized handwriting models on a **12GB laptop GPU** or **Google Colab free tier (T4 16GB)**.

**Key Findings:**
- **DeepSeek-OCR** (3B params): Perfect fit - purpose-built for OCR, handwriting-ready with LoRA fine-tuning
- **TRL's SFTTrainer**: 60% memory reduction via Liger Kernel + packing + optimizations
- **Optimized LoRA**: r=16-32, alpha=32-64 for complex handwriting (vs current r=8, alpha=16)
- **QLoRA (4-bit)**: Enables 3B-7B models on 12GB VRAM with minimal quality loss
- **Recommended dataset**: 50-200 samples for personalization (current: unlimited)

**Current State Issues:**
1. Training orchestration: Messy mix of bash scripts, Ray actors, manual CLI
2. Memory inefficiency: Not using TRL optimizations (60% memory savings on table)
3. Missing DeepSeek-OCR: The best model for this task isn't integrated
4. No Colab support: Excludes 90% of potential users
5. LoRA undertuned: r=8 works but r=16-32 would capture handwriting complexity better
6. No systematic benchmarking: Can't compare models objectively

---

## Research Findings

### 1. DeepSeek-OCR Analysis

**Architecture:**
- 3B parameter vision-language model (BF16)
- Built specifically for OCR with document-to-markdown conversion
- "Contexts optical compression" - efficient visual-text processing
- Multiple size configurations: Tiny → Large (deployment flexibility)

**Memory Profile (estimated):**
- Full precision (BF16): ~6GB VRAM
- 4-bit quantized (QLoRA): ~2.4GB VRAM
- With LoRA (r=16): +~800MB → **~3.2GB total**
- **Safe for 12GB GPU with batch_size=2-3**

**Handwriting Recommendations (from research):**
- For noisy/handwritten: Light LoRA fine-tune with 50-200 image→text pairs
- Freeze base model, use rank 8-16
- Train 1-2 epochs with early stopping
- Data augmentation: ±2° deskew, mild blur, JPEG quality jitter
- **Rule of thumb: Prompt engineering for clean layouts, LoRA for cursive/messy handwriting**

**Advantages:**
- Purpose-built for OCR (vs general VLMs like SmolVLM/Qwen3)
- Smaller than Qwen3-7B, larger than SmolVLM-256M (sweet spot)
- Supports vLLM for fast inference
- Active development (released Oct 2025)

**Disadvantages:**
- No official fine-tuning examples yet (GitHub issue #43 requesting LoRA examples)
- Less community adoption than Qwen/SmolVLM
- Need to test handwriting performance vs IAM-trained models

### 2. TRL (Transformer Reinforcement Learning)

**What it is:**
- HuggingFace library for post-training (SFT, DPO, GRPO, PPO, etc.)
- **SFTTrainer**: Drop-in replacement for HF Trainer with quality-of-life improvements
- Native PEFT integration, vLLM support, distributed training (DDP, DeepSpeed, FSDP)

**Key advantages over vanilla Trainer:**

1. **Memory Optimizations:**
   - **Liger Kernel**: 60% memory reduction, 20% throughput increase (Triton kernels for LLM training)
   - **Packing**: Flattens batches into single sequences, eliminates padding waste
   - **Activation offloading**: Offload activations to CPU during forward pass
   - **Sequence truncation**: Smart truncation to reduce memory
   - **QLoRA integration**: Seamless 4-bit quantization + LoRA

2. **Dataset Formatting:**
   - Built-in conversational and instruction format support
   - Handles chat templates automatically
   - Better than manual tokenization in current code

3. **Training Efficiency:**
   - AdamW8bit: 25-30% VRAM reduction (negligible quality loss)
   - Gradient checkpointing by default
   - Smart batching strategies

**Memory Comparison (7B model on 16GB GPU):**
- Vanilla Trainer: OOM or batch_size=1
- TRL + QLoRA + Liger Kernel: batch_size=4-8
- **60% memory reduction = 3x larger effective batch size**

### 3. LoRA Best Practices for Vision-Language Models

**Rank (r) Guidelines:**
- **r=8-16**: Simple patterns, clean handwriting
- **r=16-32**: Complex handwriting, cursive, multiple styles (recommended for DisgraPhi)
- **r=32-64**: Maximum adaptability, highest quality (diminishing returns)
- Higher rank = more parameters = larger adapter = more overfitting risk

**Alpha Guidelines:**
- **alpha = rank**: Most predictable, stable results
- **alpha = 2×rank**: Sweet spot for faster convergence (e.g., r=16, alpha=32)
- **alpha < rank**: Better detail capture, requires careful LR tuning

**Current vs Recommended:**
```
Current:  r=8,  alpha=16  (conservative, works but undertrained)
Proposed: r=16, alpha=32  (captures handwriting complexity better)
Advanced: r=32, alpha=64  (maximum quality, watch for overfitting)
```

**Target Modules:**
Current code targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
✅ This is correct - covers attention + MLP layers

**Vision Layer Considerations:**
- For handwriting: `finetune_vision_layers=False, finetune_language_layers=True`
- Vision encoder learns general features (lines, curves)
- Language decoder learns character mappings
- **Current code doesn't distinguish - should add this option**

### 4. Memory Optimization Techniques for 12GB VRAM

**Layered Strategy (apply all for maximum efficiency):**

1. **Quantization (QLoRA):**
   - 4-bit NF4 quantization: 75% memory reduction
   - `bnb_4bit_use_double_quant=True`: Additional 5-10% savings
   - Current code ✅ already implements this

2. **LoRA Configuration:**
   - r=16: ~2M trainable params (vs 3B frozen)
   - 0.07% of model parameters trainable
   - Adapter size: ~50-200MB (easily shareable)

3. **Optimizer:**
   - AdamW8bit: 25-30% VRAM reduction
   - Current code uses AdamW (32-bit) - **should migrate to 8-bit**

4. **Batch Size + Gradient Accumulation:**
   - Current: `batch_size=1, grad_accum=4` → effective=4
   - Optimized: `batch_size=2-3, grad_accum=4` → effective=8-12
   - **Can likely increase batch_size with TRL optimizations**

5. **Gradient Checkpointing:**
   - Current code ✅ already uses this
   - Trades compute for memory (20-30% slower, 40-60% less memory)

6. **Mixed Precision:**
   - BF16 for A100/4090/4080 ✅ (current code)
   - FP16 for older GPUs
   - GradScaler for stability ✅ (current code)

7. **Liger Kernel (NEW):**
   - 60% memory reduction
   - 20% throughput increase
   - Only available in TRL, not vanilla Trainer

**Estimated VRAM Budget (DeepSeek-OCR 3B + QLoRA):**
```
Model (4-bit):           ~2.4 GB
LoRA adapters (r=16):    ~0.8 GB
Optimizer states (8bit): ~1.5 GB
Activations (batch=2):   ~2.5 GB
Gradients:               ~1.8 GB
Buffer:                  ~1.0 GB
--------------------------------
Total:                   ~10.0 GB (fits in 12GB with 2GB headroom)

With Liger Kernel (-60%): ~4.0 GB (massive headroom for batch=4-6)
```

### 5. Google Colab Free Tier Optimization

**Hardware:**
- GPU: NVIDIA T4 (16GB VRAM)
- RAM: 12GB system RAM
- Disk: ~100GB temporary storage
- Runtime: 12-hour max session
- GPU availability: Not guaranteed, peak hours = no GPU

**Optimization Strategies:**

1. **Checkpointing to Google Drive:**
   - Save every N steps to persistent storage
   - Resume from last checkpoint on disconnect
   - Auto-save final adapter

2. **Dataset Caching:**
   - Download IAM → cache to Drive (first run only)
   - Subsequent runs load from cache (saves 10-15 min)

3. **Memory Management:**
   - QLoRA mandatory (4-bit)
   - AdamW8bit optimizer
   - batch_size=2-4 (T4 has 16GB, more headroom than 12GB laptop)
   - Clear cache between experiments

4. **Time Management:**
   - Bootstrap training (13k samples, 3 epochs): ~3-5 hours
   - Personalization (100 samples, 2 epochs): ~15-30 minutes
   - **Total: fits in 12-hour window with margin**

5. **User Experience:**
   - One-click setup cells
   - Progress bars for long operations
   - Auto-upload adapters to HuggingFace
   - Clear error messages with recovery steps

### 6. Training Orchestration Analysis

**Current State (MESSY):**

1. **ml/models/train.py** (Manual CLI):
   - Direct HuggingFace Trainer
   - Good: Clean, well-documented
   - Bad: Manual parameter tuning, no memory optimizations

2. **ml/ray_train.py** (Ray Distributed):
   - Ray actors for parallel training
   - Good: Smart VRAM packing, auto-upload
   - Bad: Overengineered for single GPU, complex for users

3. **ml/train_all_bootstrap.sh** (Bash script):
   - Parallel training with MPS
   - Good: Simple, works
   - Bad: Hardcoded for A100, not configurable, poor error handling

**Problems:**
- Three different systems doing similar things
- No unified config
- Hard to switch between laptop/Colab/cloud
- No experiment comparison framework
- User must understand all three systems

**Proposed Solution: Unified Config-Driven System**

**Single entrypoint:** `ml/train.py`

**Config files:** `ml/configs/{laptop_12gb,colab_t4,a100_40gb,personalize}.yaml`

**Example config:**
```yaml
# ml/configs/laptop_12gb.yaml
hardware:
  device: cuda
  vram_gb: 12
  optimization: aggressive  # aggressive, balanced, conservative

model:
  provider: deepseek-ocr  # or smolvlm-256m, qwen3-vl-2b
  quantization: 4bit
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: auto  # or list
    finetune_vision: false

training:
  mode: bootstrap  # or personalize
  dataset: iam
  sample_ratio: 1.0
  epochs: 3
  batch_size: auto  # calculates based on VRAM
  grad_accum: auto
  learning_rate: 2e-4
  optimizer: adamw8bit
  scheduler: cosine_with_warmup
  warmup_ratio: 0.1

memory:
  use_liger_kernel: true
  use_packing: true
  gradient_checkpointing: true
  activation_offloading: false

checkpointing:
  save_every: 500
  keep_best: 3
  metric: eval_loss
  upload_to_hf: true
  hf_org: velocitatem

logging:
  tensorboard: true
  log_every: 10
  eval_every: 100
```

**Benefits:**
- One command: `python ml/train.py --config laptop_12gb`
- Easy to share configs
- Automatic VRAM calculation
- Clear separation of concerns
- Easy to add new hardware profiles

---

## Implementation Plan

### Phase 1: DeepSeek-OCR Integration & Benchmarking
**Goal:** Add DeepSeek-OCR as a model provider and benchmark it

**Tasks:**
- [ ] Create `ml/models/providers/deepseek_ocr.py` implementing `BaseVisionLanguageModel`
- [ ] Add DeepSeek-OCR to model factory in `ml/models/providers/__init__.py`
- [ ] Test memory footprint with QLoRA on different configs:
  - [ ] 4-bit, r=8, batch=1
  - [ ] 4-bit, r=16, batch=2
  - [ ] 4-bit, r=32, batch=1
- [ ] Train bootstrap adapter on IAM (10% sample) for quick validation
- [ ] Benchmark against SmolVLM-256M and Qwen3-VL-2B:
  - [ ] CER, WER, NED on IAM test set
  - [ ] VRAM usage
  - [ ] Training speed (samples/sec)
  - [ ] Inference speed
- [ ] Create benchmark results table in `ml/BENCHMARKS.md`

**Success Criteria:**
- DeepSeek-OCR trains successfully on 12GB GPU
- CER comparable to or better than existing models
- Memory footprint under 10GB with batch_size≥2

**Estimated Time:** 2-3 days

### Phase 2: TRL Migration & Memory Optimization
**Goal:** Migrate to TRL's SFTTrainer with all memory optimizations

**Tasks:**
- [ ] Install TRL and dependencies: `pip install trl liger-kernel`
- [ ] Create `ml/models/train_trl.py` as new entrypoint
- [ ] Migrate `TrainingParams` to `SFTConfig`
- [ ] Replace `Trainer` with `SFTTrainer`
- [ ] Implement optimizations:
  - [ ] Liger Kernel integration
  - [ ] Packing/padding-free batching
  - [ ] AdamW8bit optimizer
  - [ ] Activation offloading (optional, test if needed)
- [ ] Add vision vs language layer fine-tuning option
- [ ] Update LoRA defaults to r=16, alpha=32
- [ ] A/B test memory usage:
  - [ ] Before: vanilla Trainer
  - [ ] After: TRL + all optimizations
- [ ] Benchmark training speed and quality (ensure no regression)
- [ ] Document memory savings in `RESEARCH.md`

**Success Criteria:**
- ≥50% memory reduction vs vanilla Trainer
- No quality degradation (CER within 1% of baseline)
- batch_size increased from 1 to 2-4 on 12GB GPU
- Training speed maintained or improved

**Estimated Time:** 3-4 days

### Phase 3: Unified Training Orchestration
**Goal:** Replace messy scripts with unified config-driven system

**Tasks:**
- [ ] Create config schema in `ml/configs/schema.yaml`
- [ ] Create hardware profiles:
  - [ ] `laptop_12gb.yaml`
  - [ ] `colab_t4.yaml`
  - [ ] `a100_40gb.yaml`
  - [ ] `personalize_laptop.yaml`
  - [ ] `personalize_colab.yaml`
- [ ] Implement config parser in `ml/config.py`
- [ ] Add automatic VRAM budget calculator:
  - [ ] Takes model size + quantization + LoRA config
  - [ ] Recommends batch_size + grad_accum
  - [ ] Warns if config exceeds VRAM
- [ ] Update `ml/train.py` to use config system
- [ ] Add experiment comparison script: `ml/compare_experiments.py`
  - [ ] Reads TensorBoard logs
  - [ ] Generates comparison table (CER, WER, VRAM, speed)
  - [ ] Exports to markdown
- [ ] Create simple CLI: `make train-bootstrap`, `make train-personalize`
- [ ] Deprecate `ray_train.py` and `train_all_bootstrap.sh` (archive in `ml/legacy/`)

**Success Criteria:**
- Single entrypoint for all training scenarios
- Config-driven, no more hardcoded parameters
- Automatic VRAM optimization
- Easy experiment comparison

**Estimated Time:** 4-5 days

### Phase 4: Google Colab Integration
**Goal:** Make training accessible to anyone with Colab free tier

**Tasks:**
- [ ] Create `notebooks/colab_bootstrap_training.ipynb`:
  - [ ] One-click setup (install deps, auth HF)
  - [ ] Download IAM dataset to Drive (with caching)
  - [ ] Train bootstrap adapter (3-5 hours)
  - [ ] Auto-upload to HuggingFace
  - [ ] Clear documentation, progress bars
- [ ] Create `notebooks/colab_personalization.ipynb`:
  - [ ] Upload user handwriting images
  - [ ] Create manifest.json automatically
  - [ ] Load bootstrap adapter from HF
  - [ ] Train personalized adapter (15-30 min)
  - [ ] Download adapter for local use
  - [ ] Test inference with examples
- [ ] Add Drive checkpointing:
  - [ ] Save every 500 steps to Drive
  - [ ] Resume logic if disconnected
- [ ] Create troubleshooting guide for common Colab issues
- [ ] Test on actual Colab free tier (not Pro)
- [ ] Create video tutorial (optional but recommended)

**Success Criteria:**
- Bootstrap training completes in <6 hours on T4
- Personalization completes in <30 minutes
- Works reliably on free tier (not just Pro)
- Non-technical users can follow notebook

**Estimated Time:** 3-4 days

### Phase 5: Dataset Handling & Augmentation
**Goal:** Implement recommended 50-200 sample personalization + smart augmentation

**Tasks:**
- [ ] Update `ManifestDataset` to support sample limits
- [ ] Add validation: warn if personalization dataset >200 samples
- [ ] Implement recommended augmentations for handwriting:
  - [ ] ±2° deskew/rotation
  - [ ] Mild Gaussian blur (sigma=0.5-1.0)
  - [ ] JPEG compression artifacts (quality=80-95)
  - [ ] Already have: elastic distortion, slant, baseline drift ✅
- [ ] Create augmentation presets:
  - [ ] `conservative`: Light augmentation for clean handwriting
  - [ ] `standard`: Balanced (recommended default)
  - [ ] `aggressive`: Heavy augmentation for small datasets
- [ ] Add augmentation visualization tool: `ml/data/visualize_augmentation.py`
  - [ ] Shows before/after samples
  - [ ] Helps users tune augmentation strength
- [ ] Update documentation on dataset collection best practices
- [ ] Create template packet generator (printable PDF with practice lines)

**Success Criteria:**
- Personalization works well with 50-100 samples
- Augmentation improves quality on small datasets (validated experimentally)
- Users have clear guidance on data collection

**Estimated Time:** 2-3 days

### Phase 6: Benchmarking Framework
**Goal:** Systematic model comparison and evaluation

**Tasks:**
- [ ] Create `ml/benchmark.py` script:
  - [ ] Runs all models on same test set
  - [ ] Measures: CER, WER, NED, VRAM, speed, adapter size
  - [ ] Generates comparison table + plots
- [ ] Add per-writer evaluation (IAM has writer IDs)
  - [ ] Shows model generalization across different handwriting styles
- [ ] Create leaderboard: `ml/LEADERBOARD.md`
  - [ ] Bootstrap models (trained on IAM)
  - [ ] Personalized models (IAM + user data)
- [ ] Add inference benchmarking:
  - [ ] Latency (ms per image)
  - [ ] Throughput (images/sec)
  - [ ] Memory footprint during inference
- [ ] Create visualization dashboard (optional):
  - [ ] Streamlit app showing benchmarks
  - [ ] Side-by-side predictions
  - [ ] Error analysis (character confusion matrix)

**Success Criteria:**
- Objective comparison of all models
- Clear winner for each use case (accuracy vs speed vs VRAM)
- Reproducible benchmarks

**Estimated Time:** 3-4 days

### Phase 7: Documentation & User Guide
**Goal:** Make system accessible to non-technical users

**Tasks:**
- [ ] Update `README.md` with quick start guide
- [ ] Create `docs/TRAINING_GUIDE.md`:
  - [ ] How to collect handwriting samples
  - [ ] How to train bootstrap (cloud vs local)
  - [ ] How to personalize
  - [ ] How to deploy adapter
- [ ] Create `docs/TROUBLESHOOTING.md`:
  - [ ] Common errors and fixes
  - [ ] VRAM optimization tips
  - [ ] Colab-specific issues
- [ ] Add model selection guide:
  - [ ] When to use DeepSeek-OCR vs SmolVLM vs Qwen3
  - [ ] Speed vs accuracy tradeoffs
- [ ] Create example use cases:
  - [ ] Student digitizing handwritten notes
  - [ ] Doctor transcribing prescriptions
  - [ ] Artist converting sketches to vector
- [ ] Add FAQ section
- [ ] Create contribution guide for new model providers

**Success Criteria:**
- Complete, clear documentation
- Non-technical users can train models
- All common issues documented

**Estimated Time:** 2-3 days

---

## Detailed Technical Specifications

### Config System Architecture

**File Structure:**
```
ml/
├── configs/
│   ├── schema.yaml           # JSON schema for validation
│   ├── hardware/
│   │   ├── laptop_12gb.yaml
│   │   ├── colab_t4.yaml
│   │   ├── a100_40gb.yaml
│   ├── models/
│   │   ├── deepseek_ocr.yaml
│   │   ├── smolvlm_256m.yaml
│   │   ├── qwen3_vl_2b.yaml
│   ├── training/
│   │   ├── bootstrap.yaml
│   │   ├── personalize.yaml
│   ├── presets/
│   │   ├── quick_test.yaml   # 10% data, 1 epoch
│   │   ├── production.yaml   # 100% data, 3 epochs
├── config.py                 # Config loader & validator
├── train.py                  # Unified entrypoint
```

**Config Composition:**
```bash
# User can combine configs
python ml/train.py \
  --hardware configs/hardware/laptop_12gb.yaml \
  --model configs/models/deepseek_ocr.yaml \
  --training configs/training/bootstrap.yaml \
  --preset configs/presets/quick_test.yaml

# Or use a single merged config
python ml/train.py --config configs/presets/laptop_bootstrap_deepseek.yaml

# Or override specific values
python ml/train.py --config laptop_12gb --epochs 5 --batch-size 4
```

**VRAM Budget Calculator:**
```python
def calculate_vram_budget(model_size_b, quantization, lora_r, batch_size):
    """
    Calculate estimated VRAM usage.

    Returns: {
        'model_gb': float,
        'optimizer_gb': float,
        'activations_gb': float,
        'gradients_gb': float,
        'total_gb': float,
        'fits_in_vram': bool,
        'recommended_batch_size': int
    }
    """
    quant_multiplier = {'4bit': 0.25, '8bit': 0.5, 'bf16': 2.0, 'fp32': 4.0}

    model_gb = model_size_b * quant_multiplier[quantization]
    lora_gb = (lora_r * model_size_b * 0.0001)  # Rough estimate
    optimizer_gb = lora_gb * 2  # 8-bit optimizer
    activations_gb = model_size_b * 0.3 * batch_size  # Rough heuristic
    gradients_gb = lora_gb * batch_size
    buffer_gb = 1.0

    total_gb = model_gb + lora_gb + optimizer_gb + activations_gb + gradients_gb + buffer_gb
    return total_gb
```

### TRL Migration: Key Changes

**Before (vanilla Trainer):**
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    # ... standard args
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
```

**After (TRL SFTTrainer):**
```python
from trl import SFTTrainer, SFTConfig
from liger_kernel.transformers import apply_liger_kernel_to_llama  # or qwen/deepseek

# Apply Liger Kernel optimizations
apply_liger_kernel_to_llama()

training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Can increase with optimizations

    # TRL-specific optimizations
    packing=True,  # Padding-free batching
    dataset_text_field="text",  # Field containing text
    max_seq_length=512,

    # Memory optimizations
    optim="adamw_8bit",  # 8-bit optimizer
    gradient_checkpointing=True,

    # ... standard args
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)
```

**Dataset Format for Packing:**
Current code uses dict with `{'image': PIL.Image, 'text': str}`. TRL expects text field for packing. Need adapter:

```python
class PackableDataset:
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __getitem__(self, idx):
        item = self.base[idx]
        # Serialize image path and text together
        return {
            'text': f"<image>{item['image_path']}</image>{item['text']}",
            'image': item['image']  # Keep for processor
        }
```

### DeepSeek-OCR Provider Implementation

**Key Implementation Details:**

```python
# ml/models/providers/deepseek_ocr.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseVisionLanguageModel

class DeepSeekOCR(BaseVisionLanguageModel):
    def __init__(self, lora_r=16, lora_alpha=32, lora_dropout=0.05,
                 load_in_4bit=True, load_in_8bit=False,
                 bootstrap_adapter_path=None):
        super().__init__()

        # Load model with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if load_in_4bit else None

        self.model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-OCR",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-OCR",
            trust_remote_code=True
        )

        # Apply LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)

        # Load bootstrap adapter if provided
        if bootstrap_adapter_path:
            self.load_adapter(bootstrap_adapter_path, "bootstrap", is_trainable=False)

    def prepare_training_batch(self, images, texts):
        # DeepSeek-OCR format: "<image>\nFree OCR."
        prompts = [f"<image>\nTranscribe: " for _ in texts]

        # Process inputs (need to check DeepSeek-OCR's processor API)
        inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True
        )

        # Create labels (shift left for causal LM)
        labels = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True
        ).input_ids

        labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)

        return {
            'pixel_values': inputs['pixel_values'].to(self.device),
            'input_ids': inputs['input_ids'].to(self.device),
            'attention_mask': inputs['attention_mask'].to(self.device),
            'labels': labels.to(self.device)
        }
```

**Testing Plan:**
1. Test import and model loading
2. Test forward pass with dummy data
3. Test LoRA adapter saving/loading
4. Test training on small subset (10 samples)
5. Test full bootstrap training (10% IAM)
6. Benchmark on IAM test set

---

## Risk Assessment & Mitigation

### Risk 1: DeepSeek-OCR may not work for handwriting
**Likelihood:** Medium
**Impact:** High (core assumption)

**Mitigation:**
- Early validation: Train on 10% IAM before full implementation
- Have fallback: Keep SmolVLM/Qwen3 as proven alternatives
- If DeepSeek-OCR underperforms: Document findings, use as secondary option
- Quick pivot threshold: If CER >15% on IAM validation, deprioritize

### Risk 2: TRL migration breaks existing workflow
**Likelihood:** Low
**Impact:** Medium

**Mitigation:**
- Keep `train.py` as legacy backup
- Create `train_trl.py` as new system
- Parallel testing: Run both systems on same data, compare outputs
- Gradual migration: Keep both until TRL proven stable

### Risk 3: Memory optimizations don't deliver promised savings
**Likelihood:** Low-Medium
**Impact:** Medium

**Mitigation:**
- Measure before/after with `torch.cuda.memory_summary()`
- Document actual savings vs claimed (manage expectations)
- If <30% savings: Still document, highlight what works
- Liger Kernel is experimental: May not work on all GPUs (test on 4080)

### Risk 4: Colab free tier limitations prevent training
**Likelihood:** Medium
**Impact:** High (accessibility goal)

**Mitigation:**
- Test extensively on actual free tier (not Pro)
- Design for interruptions: Aggressive checkpointing
- Provide fallback: Cloud options (Vast.ai, RunPod) if Colab fails
- Document realistic expectations: 12-hour limit, GPU availability

### Risk 5: User-collected datasets are too noisy/inconsistent
**Likelihood:** High
**Impact:** Medium

**Mitigation:**
- Provide detailed data collection guide
- Create validation script: Checks image quality, detects errors
- Add data cleaning utilities (crop, deskew, contrast adjust)
- Template packet generator: Standardized format for consistency

---

## Success Metrics

### Technical Metrics:
- **Memory Efficiency:** ≥50% VRAM reduction vs current implementation
- **Training Speed:** No regression (±10% acceptable)
- **Model Quality:** CER ≤5% on IAM test set (bootstrap), ≤3% on personalized
- **Accessibility:** Training completes on 12GB GPU with batch_size≥2

### User Experience Metrics:
- **Setup Time:** <10 minutes from clone to first training run (Colab)
- **Bootstrap Training:** <6 hours on Colab free tier
- **Personalization:** <30 minutes with 100 samples
- **Documentation:** Non-technical user can follow guide without errors

### Ecosystem Metrics:
- **Model Availability:** All bootstrap adapters on HuggingFace
- **Community Adoption:** 10+ external users train personalized models
- **Reproducibility:** Other researchers can replicate benchmarks

---

## Timeline & Resource Estimate

**Total Estimated Time:** 19-27 days (4-6 weeks)

**Phase Breakdown:**
- Phase 1 (DeepSeek-OCR): 2-3 days
- Phase 2 (TRL Migration): 3-4 days
- Phase 3 (Unified Config): 4-5 days
- Phase 4 (Colab): 3-4 days
- Phase 5 (Datasets): 2-3 days
- Phase 6 (Benchmarking): 3-4 days
- Phase 7 (Documentation): 2-3 days
- Buffer for bugs/revisions: 3-5 days

**Critical Path:**
1. Phase 1 → Phase 2 (need DeepSeek working before TRL migration)
2. Phase 2 → Phase 3 (need TRL working before config system)
3. Phase 3 → Phase 4 (need config system before Colab notebooks)
4. Phases 5-6-7 can run in parallel after Phase 3

**Resource Requirements:**
- **Compute:**
  - 12GB GPU for development/testing (laptop 4080)
  - Colab free tier for validation
  - Optional: Cloud GPU for final benchmarks (~$20-50)
- **Storage:**
  - IAM dataset: ~5GB
  - Model checkpoints: ~50-100GB (bootstrap + experiments)
  - HuggingFace storage: Free for public models
- **Human Time:**
  - Full-time: 4-6 weeks
  - Part-time (20hr/week): 10-14 weeks

---

## Next Steps

### Immediate Actions (Week 1):
1. ✅ Research TRL, LoRA, DeepSeek-OCR (DONE)
2. ✅ Create this implementation plan (DONE)
3. **Review plan with stakeholders** (if applicable)
4. **Set up development environment:**
   - Install TRL: `pip install trl liger-kernel`
   - Test on 12GB GPU (laptop 4080)
   - Verify Colab access

### Week 2-3: Phase 1 + Phase 2
- Implement DeepSeek-OCR provider
- Validate on IAM subset
- Migrate to TRL
- Measure memory improvements

### Week 4-5: Phase 3 + Phase 4
- Build config system
- Create Colab notebooks
- Test end-to-end on free tier

### Week 6+: Phase 5-7 + Polish
- Dataset handling improvements
- Benchmarking framework
- Documentation
- Final testing and release

---

## Open Questions & Future Research

1. **DeepSeek-OCR fine-tuning API:** Official examples don't exist yet. May need to reverse-engineer from model architecture.

2. **Optimal LoRA rank for handwriting:** Research suggests r=16-32, but need empirical validation on IAM. Plan A/B tests.

3. **Vision layer fine-tuning:** Should we fine-tune vision encoder or freeze it? Current code doesn't distinguish.

4. **Multi-adapter composition:** PEFT supports stacking adapters. Could we have base OCR + style-specific adapters?

5. **Continuous learning:** Can users incrementally improve adapters with new samples without catastrophic forgetting?

6. **Synthetic data:** Could we generate synthetic handwriting to augment small user datasets?

7. **Few-shot learning:** What's the minimum viable dataset? 50? 20? 10 samples?

8. **Transfer learning:** Bootstrap on IAM (English) → personalize on other languages?

---

## Conclusion

This plan transforms DisgraPhi from a research project into an accessible, production-ready system for personalized handwriting recognition. Key innovations:

1. **DeepSeek-OCR:** Purpose-built OCR model (3B params, perfect for handwriting)
2. **TRL + Liger Kernel:** 60% memory reduction, enables larger models on 12GB GPU
3. **Unified config system:** Replace messy scripts with clean, composable configs
4. **Colab support:** Democratize access, anyone can train models
5. **Systematic benchmarking:** Objective model comparison, reproducible science

**Philosophy:** Make the complex simple. Hide the technical complexity behind clean interfaces, clear documentation, and smart defaults. Users should think about their handwriting, not VRAM budgets.

**Validation Strategy:** Fail fast. Test core assumptions (DeepSeek-OCR performance, TRL memory savings) in Phase 1-2 before building on top. If something doesn't work, pivot quickly.

**Success Looks Like:**
- Student uploads 100 handwriting samples → 30 minutes later → personalized OCR model
- Researcher replicates benchmarks with one command
- Developer adds new model provider in <200 lines of code
- Community shares adapters on HuggingFace

Let's build it.
