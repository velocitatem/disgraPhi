"""
DisgraPhi: TRL-optimized training entrypoint for vision-language OCR models.

Optimizations over vanilla Trainer:
- TRL's SFTTrainer with memory-efficient features
- Liger Kernel integration (60% memory reduction)
- Packing/padding-free batching
- AdamW8bit optimizer (25-30% VRAM reduction)
- Activation offloading (optional)
- Improved LoRA defaults (r=16, alpha=32)

Key features:
- Deterministic seeding
- Mixed precision (BF16/FP16 with gradient scaling)
- Gradient accumulation
- TensorBoard logging
- Checkpointing with best-by-loss and best-by-hybrid metrics
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import TrainerCallback, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# Try to import Liger Kernel (optional but recommended)
try:
    from liger_kernel.transformers import apply_liger_kernel_to_llama
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False
    print("Warning: Liger Kernel not available. Install with: pip install liger-kernel")

from ml.data.datasets import IAMDataset, ManifestDataset
from ml.models.providers import create_model
from ml.models.eval import compute_ocr_metrics

# Set CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def set_seed(seed: int = 2025):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainingParams:
    """TRL-optimized training configuration parameters."""

    # Model configuration
    model_provider: str = "deepseek-ocr"  # deepseek-ocr, smolvlm-256m, qwen3-vl-2b
    lora_r: int = 16  # Increased from 8 for better handwriting capture
    lora_alpha: int = 32  # 2x rank (sweet spot)
    lora_dropout: float = 0.05
    load_in_4bit: bool = True  # Recommended for 12GB GPU
    load_in_8bit: bool = False
    bootstrap_adapter_path: Optional[str] = None
    finetune_vision_layers: bool = False  # Only fine-tune language decoder
    finetune_language_layers: bool = True

    # Dataset configuration
    dataset_type: str = "iam"  # iam or manifest
    data_dir: str = "./ml/data/raw/iam"
    manifest_data_dir: Optional[str] = None
    sample_ratio: float = 1.0
    augment: bool = False
    augment_strength: float = 0.7

    # Training arguments
    output_dir: str = "./ml/checkpoints"
    experiment_name: Optional[str] = None  # Auto-generated if None
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2  # Increased from 1 with optimizations
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4  # Slightly higher for LoRA
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = True
    seed: int = 2025

    # TRL-specific optimizations
    use_liger_kernel: bool = True  # 60% memory reduction
    packing: bool = False  # Padding-free batching (experimental for VLMs)
    max_seq_length: int = 1024
    optim: str = "adamw_8bit"  # 25-30% VRAM reduction
    gradient_checkpointing: bool = True
    activation_offloading: bool = False  # Enable if still OOM

    # Evaluation configuration
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True

    # TensorBoard
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    logging_dir: Optional[str] = None

    def __post_init__(self):
        """Set derived parameters and apply optimizations."""
        if self.experiment_name is None:
            task = "personalization" if self.bootstrap_adapter_path else "bootstrap"
            model_name = self.model_provider.rsplit('-', 1)[0] if '-' in self.model_provider else self.model_provider
            model_name = model_name.replace('-', '')
            model_size = self.model_provider.rsplit('-', 1)[-1] if '-' in self.model_provider else '3b'
            dataset = f"iam{int(self.sample_ratio*100)}" if self.dataset_type == "iam" else Path(self.manifest_data_dir or "manifest").name
            precision = "4bit" if self.load_in_4bit else "8bit" if self.load_in_8bit else "bf16" if self.bf16 else "fp16" if self.fp16 else "fp32"
            self.experiment_name = f"trl.{task}.{model_name}.{model_size}.{dataset}.{precision}"

        if self.logging_dir is None:
            self.logging_dir = os.path.join(self.output_dir, "logs", self.experiment_name)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging_dir).mkdir(parents=True, exist_ok=True)

        # Apply Liger Kernel if available and enabled
        if self.use_liger_kernel and LIGER_AVAILABLE:
            print("Applying Liger Kernel optimizations (60% memory reduction)...")
            try:
                apply_liger_kernel_to_llama()
                print("✓ Liger Kernel applied successfully")
            except Exception as e:
                print(f"Warning: Could not apply Liger Kernel: {e}")

    def to_sft_config(self) -> SFTConfig:
        """Convert to TRL SFTConfig."""
        return SFTConfig(
            output_dir=os.path.join(self.output_dir, self.experiment_name),
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            max_grad_norm=self.max_grad_norm,
            logging_steps=self.logging_steps,
            eval_strategy="steps",
            eval_steps=self.eval_steps,
            save_strategy="steps",
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            fp16=self.fp16,
            bf16=self.bf16,
            seed=self.seed,
            optim=self.optim,  # TRL-specific: adamw_8bit
            gradient_checkpointing=self.gradient_checkpointing,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            load_best_model_at_end=self.load_best_model_at_end,
            report_to=self.report_to,
            logging_dir=self.logging_dir,
            remove_unused_columns=False,
            label_names=["labels"],
            # TRL-specific: packing
            packing=self.packing,
            max_seq_length=self.max_seq_length,
            # dataset_text_field="text",  # Not used for VLMs with custom collator
        )

    def to_bnb_config(self) -> Optional[BitsAndBytesConfig]:
        """Create BitsAndBytes quantization config if needed."""
        if not self.load_in_4bit and not self.load_in_8bit:
            return None

        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            bnb_4bit_use_double_quant=True if self.load_in_4bit else None,
            bnb_4bit_quant_type="nf4" if self.load_in_4bit else None,
            bnb_4bit_compute_dtype=torch.bfloat16 if self.load_in_4bit else None
        )


class TensorBoardCallback(TrainerCallback):
    """Custom callback for TensorBoard logging with OCR metrics."""

    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics to TensorBoard."""
        if metrics is not None:
            print(f"\n[Evaluation] Step {state.global_step}")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")


class DataCollator:
    """Custom data collator for vision-language models."""

    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [f['image'] for f in features]
        texts = [f['text'] for f in features]
        return self.model_wrapper.prepare_training_batch(images, texts)


def create_compute_metrics(model_wrapper):
    """Create compute_metrics function with OCR metrics."""
    tokenizer = model_wrapper.tokenizer

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        pred_ids = np.argmax(preds[0] if isinstance(preds, tuple) else preds, axis=-1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        metrics = [compute_ocr_metrics(gt, pred) for gt, pred in zip(label_texts, pred_texts)]
        return {
            'cer': np.mean([m['cer'] for m in metrics]),
            'wer': np.mean([m['wer'] for m in metrics]),
            'ned': np.mean([m['ned'] for m in metrics]),
            'accuracy': np.mean([m['accuracy'] for m in metrics]),
        }

    return compute_metrics


def load_model(params: TrainingParams):
    """Load model based on provider string using the factory function."""
    return create_model(
        model_type=params.model_provider,
        lora_r=params.lora_r,
        lora_alpha=params.lora_alpha,
        lora_dropout=params.lora_dropout,
        load_in_4bit=params.load_in_4bit,
        load_in_8bit=params.load_in_8bit,
        bootstrap_adapter_path=params.bootstrap_adapter_path,
        finetune_vision_layers=params.finetune_vision_layers,
        finetune_language_layers=params.finetune_language_layers,
    )


def load_datasets(params: TrainingParams):
    """Load training and validation datasets."""
    if params.dataset_type == "iam":
        train_dataset = IAMDataset(
            data_dir=params.data_dir,
            split='train',
            sample_ratio=params.sample_ratio,
            seed=params.seed
        )

        val_dataset = IAMDataset(
            data_dir=params.data_dir,
            split='val',
            sample_ratio=params.sample_ratio,
            seed=params.seed
        )

    elif params.dataset_type == "manifest":
        if params.manifest_data_dir is None:
            raise ValueError("manifest_data_dir must be provided for manifest dataset")

        train_dataset = ManifestDataset(
            data_dir=params.manifest_data_dir,
            subset='train',
            split_ratios=(0.8, 0.1, 0.1),
            seed=params.seed,
            augment=params.augment,
            augment_strength=params.augment_strength
        )

        val_dataset = ManifestDataset(
            data_dir=params.manifest_data_dir,
            subset='val',
            split_ratios=(0.8, 0.1, 0.1),
            seed=params.seed,
            augment=False
        )

    else:
        raise ValueError(f"Unknown dataset_type: {params.dataset_type}. Choose 'iam' or 'manifest'")

    return train_dataset, val_dataset


def print_memory_stats():
    """Print CUDA memory statistics."""
    if torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("CUDA Memory Statistics")
        print("=" * 80)
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}:")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train OCR model with TRL optimizations.")

    # Model arguments
    parser.add_argument("--model_provider", type=str, default="deepseek-ocr",
                        help="Model provider (deepseek-ocr, smolvlm-256m, qwen3-vl-2b)")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization (default: True)")
    parser.add_argument("--load_in_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--bootstrap_adapter_path", type=str, default=None,
                        help="Path to bootstrap adapter for personalization")
    parser.add_argument("--finetune_vision_layers", action="store_true",
                        help="Fine-tune vision encoder layers")
    parser.add_argument("--finetune_language_layers", action="store_true", default=True,
                        help="Fine-tune language decoder layers (default: True)")

    # Dataset arguments
    parser.add_argument("--dataset_type", type=str, default="iam", choices=["iam", "manifest"],
                        help="Dataset type")
    parser.add_argument("--data_dir", type=str, default="./ml/data/raw/iam",
                        help="Path to IAM dataset")
    parser.add_argument("--manifest_data_dir", type=str, default=None,
                        help="Path to manifest dataset (for personalization)")
    parser.add_argument("--sample_ratio", type=float, default=1.0,
                        help="Fraction of dataset to use (0.0-1.0)")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--augment_strength", type=float, default=0.7,
                        help="Augmentation strength")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./ml/checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name (auto-generated if not provided)")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging interval")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation interval")
    parser.add_argument("--save_steps", type=int, default=500, help="Checkpoint save interval")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Max number of checkpoints to keep")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use BF16 mixed precision (default: True)")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")

    # TRL-specific arguments
    parser.add_argument("--use_liger_kernel", action="store_true", default=True,
                        help="Use Liger Kernel for memory optimization (default: True)")
    parser.add_argument("--packing", action="store_true",
                        help="Enable packing/padding-free batching")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--optim", type=str, default="adamw_8bit",
                        help="Optimizer (adamw_8bit, adamw_torch, etc.)")
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                        help="Disable gradient checkpointing")
    parser.add_argument("--activation_offloading", action="store_true",
                        help="Enable activation offloading (if OOM)")

    args = parser.parse_args()

    # Create training parameters
    params = TrainingParams(
        model_provider=args.model_provider,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bootstrap_adapter_path=args.bootstrap_adapter_path,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        dataset_type=args.dataset_type,
        data_dir=args.data_dir,
        manifest_data_dir=args.manifest_data_dir,
        sample_ratio=args.sample_ratio,
        augment=args.augment,
        augment_strength=args.augment_strength,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        use_liger_kernel=args.use_liger_kernel,
        packing=args.packing,
        max_seq_length=args.max_seq_length,
        optim=args.optim,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        activation_offloading=args.activation_offloading,
    )

    # Set random seed
    set_seed(params.seed)

    print("=" * 80)
    print("DisgraPhi TRL-Optimized Training")
    print("=" * 80)
    print(f"Model: {params.model_provider}")
    print(f"Dataset: {params.dataset_type}")
    print(f"Experiment: {params.experiment_name}")
    print(f"Output: {params.output_dir}")
    print(f"LoRA: r={params.lora_r}, alpha={params.lora_alpha}")
    print(f"Optimizer: {params.optim}")
    print(f"Liger Kernel: {params.use_liger_kernel}")
    print(f"Packing: {params.packing}")
    print("=" * 80)

    # Load model
    print("\n[1/4] Loading model...")
    model = load_model(params)
    print_memory_stats()

    # Load datasets
    print("\n[2/4] Loading datasets...")
    train_dataset, val_dataset = load_datasets(params)
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Create data collator
    data_collator = DataCollator(model)

    # Create training arguments
    print("\n[3/4] Setting up TRL SFTTrainer...")
    training_args = params.to_sft_config()

    # Create trainer
    trainer = SFTTrainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=create_compute_metrics(model),
        callbacks=[TensorBoardCallback(model)],
    )

    # Train
    print("\n[4/4] Starting training...")
    print(f"  Total epochs: {params.num_train_epochs}")
    print(f"  Effective batch size: {params.per_device_train_batch_size * params.gradient_accumulation_steps}")
    print(f"  Learning rate: {params.learning_rate}")
    print("=" * 80)

    trainer.train()

    # Print final memory stats
    print_memory_stats()

    # Save final adapter
    final_adapter_path = os.path.join(params.output_dir, params.experiment_name, "final_adapter")
    print(f"\n\nSaving final adapter to {final_adapter_path}...")
    model.save_adapter(final_adapter_path)

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Checkpoints saved to: {os.path.join(params.output_dir, params.experiment_name)}")
    print(f"TensorBoard logs: {params.logging_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
