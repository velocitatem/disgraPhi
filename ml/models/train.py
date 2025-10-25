"""
DisgraPhi Training Loop

Unified training for all vision-language models:
- Qwen3-VL (4B/7B)
- SmolVLM (256M/500M/2.2B)
- Future models...

Two-stage training:
1. Bootstrap: Train on IAM database (~115k lines)
2. Personalization: Fine-tune on user data (40-120 lines)

Features:
- LoRA-based training with PEFT
- Tensorboard logging
- Gradient accumulation
- Mixed precision training
- Checkpoint management
- Model-agnostic via BaseVisionLanguageModel interface
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
from alveslib import get_logger

from ml.models.providers import create_model, MODEL_REGISTRY, BaseVisionLanguageModel
from ml.data.datasets import IAMDataset, PersonalizationDataset, ManifestDataset
from ml.models.eval import model_agnostic_loss


logger = get_logger("ml-trainloop")


class DisgraPhiTrainer:
    """
    Universal trainer for vision-language models.

    Works with any model implementing BaseVisionLanguageModel interface.

    Args:
        model: BaseVisionLanguageModel instance (Qwen3, SmolVLM, etc.)
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Directory to save checkpoints and logs
        experiment_name: Name for tensorboard logging
        learning_rate: Learning rate
        batch_size: Batch size per device
        gradient_accumulation_steps: Steps to accumulate gradients
        num_epochs: Number of training epochs
        warmup_steps: Number of warmup steps for lr scheduler
        log_every: Log metrics every N steps
        save_every: Save checkpoint every N steps
        eval_every: Run evaluation every N steps
    """

    def __init__(
        self,
        model: BaseVisionLanguageModel,
        train_dataset,
        val_dataset,
        output_dir: str,
        experiment_name: str = "default",
        learning_rate: float = 1e-5,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        num_epochs: int = 3,
        warmup_steps: int = 0,  # 0 means auto-calculate as 30% of total steps
        log_every: int = 10,
        save_every: int = 500,
        eval_every: int = 50,
        max_grad_norm: float = 1.0,
        # Hierarchical PEFT parameters
        hierarchical_mode: bool = False,
        kl_weight: float = 0.5,
        l2sp_weight: float = 1e-4,
        base_model_for_kl: Optional[BaseVisionLanguageModel] = None
    ):
        self.model = model
        self.experiment_name = experiment_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training config
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.log_every = log_every
        self.save_every = save_every
        self.eval_every = eval_every
        self.max_grad_norm = max_grad_norm

        # Hierarchical PEFT settings
        self.hierarchical_mode = hierarchical_mode
        self.kl_weight = kl_weight
        self.l2sp_weight = l2sp_weight
        self.base_model_for_kl = base_model_for_kl  # Model with only bootstrap adapter (no user adapter)

        if hierarchical_mode:
            logger.info("=== Hierarchical PEFT Mode Enabled ===")
            logger.info(f"  KL divergence weight: {kl_weight}")
            logger.info(f"  L2-SP weight: {l2sp_weight}")
            if base_model_for_kl is None:
                logger.warning("No base model provided for KL divergence - KL loss will be skipped")

        # Setup dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn
        )

        # Setup optimizer (only trainable parameters)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.05  # Increased from 0.01 for better regularization
        )

        # Setup lr scheduler
        num_training_steps = len(self.train_loader) * num_epochs // gradient_accumulation_steps
        # Use 30% of training steps for warmup if warmup_steps not explicitly set
        effective_warmup_steps = warmup_steps if warmup_steps > 0 else int(0.3 * num_training_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=effective_warmup_steps,
            num_training_steps=num_training_steps
        )

        # Tensorboard writer
        log_dir = self.output_dir / 'tensorboard' / experiment_name
        self.writer = SummaryWriter(log_dir=str(log_dir))

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_hybrid_loss = float('inf')

        logger.info(f"Trainer initialized")
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Validation samples: {len(val_dataset)}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
        logger.info(f"  Total steps: {num_training_steps}")
        logger.info(f"  Learning rate: {learning_rate}")

    def _collate_fn(self, batch):
        """
        Collate function for batching.

        Returns raw images and texts - let the model provider handle preprocessing.
        """
        images = [item['image'] for item in batch]
        texts = [item['text'] for item in batch]

        # Delegate preprocessing to model
        return self.model.prepare_training_batch(images, texts)

    def _compute_gradient_norm(self) -> float:
        """Compute the global gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None and p.requires_grad:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def _log_parameter_stats(self) -> Dict[str, float]:
        """Compute statistics for LoRA parameters."""
        stats = {}
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                stats[f'params/{name}/mean'] = param.data.mean().item()
                stats[f'params/{name}/std'] = param.data.std().item()
                stats[f'params/{name}/abs_max'] = param.data.abs().max().item()
        return stats

    def _compute_perplexity(self, loss: float) -> float:
        """Compute perplexity from cross-entropy loss."""
        return torch.exp(torch.tensor(loss)).item()

    def _compute_l2sp_loss(self) -> torch.Tensor:
        """
        Compute L2-SP regularization loss.

        Regularizes user adapter weights toward zero (or global adapter).
        Only applies to trainable parameters.
        """
        l2sp_loss = 0.0
        num_params = 0

        for name, param in self.model.named_parameters():
            # Only regularize trainable user adapter parameters
            # Skip bootstrap adapter (frozen) and base model
            if param.requires_grad and 'lora' in name.lower():
                l2sp_loss += torch.sum(param ** 2)
                num_params += 1

        if num_params > 0:
            l2sp_loss = l2sp_loss / num_params

        return l2sp_loss

    def _compute_kl_divergence_loss(
        self,
        batch: Dict[str, torch.Tensor],
        user_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between base model (bootstrap only) and user model.

        This regularizes the user adapter to stay close to the bootstrap model,
        preventing overfitting on few samples.

        Args:
            batch: Input batch
            user_logits: Logits from model with both bootstrap + user adapters

        Returns:
            KL divergence loss
        """
        if self.base_model_for_kl is None:
            return torch.tensor(0.0, device=user_logits.device)

        # Get base model predictions (bootstrap only, no user adapter)
        with torch.no_grad():
            base_outputs = self.base_model_for_kl(
                pixel_values=batch['pixel_values'],
                image_grid_thw=batch.get('image_grid_thw'),
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=None  # No loss computation
            )
            base_logits = base_outputs['logits']

        # Compute KL divergence: KL(P_base || P_user)
        # P_base is the teacher (bootstrap), P_user is the student (bootstrap + user)
        kl_loss = nn.functional.kl_div(
            nn.functional.log_softmax(user_logits, dim=-1),
            nn.functional.softmax(base_logits, dim=-1),
            reduction='batchmean'
        )

        return kl_loss

    def _generate_sample_predictions(
        self,
        num_samples: int = 3
    ) -> List[Dict[str, str]]:
        """Generate sample predictions for qualitative evaluation."""
        self.model.eval()
        samples = []

        # Get a few samples from validation set
        val_samples = []
        for i, sample in enumerate(self.val_dataset):
            if i >= num_samples:
                break
            val_samples.append(sample)

        with torch.no_grad():
            for sample in val_samples:
                image = sample['image']
                ground_truth = sample['text']

                try:
                    # Convert PIL image to RGB if needed
                    if hasattr(image, 'convert'):
                        image = image.convert('RGB')

                    # Use model's generate method (unified interface)
                    prediction = self.model.generate(
                        pixel_values=image,
                        prompt="Transcribe this handwritten text.",
                        max_new_tokens=128,
                        temperature=0.0  # Greedy decoding for eval
                    )

                    # Handle models that return tuple (text, parsed_dict)
                    # vs models that return just text
                    if isinstance(prediction, tuple):
                        prediction_text = prediction[0]  # Extract just the text
                    else:
                        prediction_text = prediction

                    samples.append({
                        'ground_truth': ground_truth,
                        'prediction': prediction_text
                    })

                except Exception as e:
                    logger.warning(f"Failed to generate sample prediction: {e}")
                    continue

        return samples

    def _compute_hybrid_ocr_loss(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """
        Compute hybrid OCR loss combining CER, WER, and NED.

        Returns average hybrid loss across all samples, plus individual metrics.
        """
        logger.info(f"Computing hybrid OCR metrics for {len(predictions)} predictions")

        # Filter out empty strings
        valid_pairs = [
            (pred, gt) for pred, gt in zip(predictions, ground_truths)
            if pred.strip() and gt.strip()
        ]

        logger.info(f"Valid pairs after filtering: {len(valid_pairs)}/{len(predictions)}")

        if not valid_pairs:
            logger.warning("No valid prediction pairs found! All predictions or ground truths are empty.")
            logger.warning(f"Sample predictions: {predictions[:3]}")
            logger.warning(f"Sample ground truths: {ground_truths[:3]}")
            return {'hybrid_loss': None}

        # Compute hybrid loss for each pair
        hybrid_losses = []
        for pred, gt in valid_pairs:
            try:
                loss = model_agnostic_loss(gt, pred)
                hybrid_losses.append(loss)
            except Exception as e:
                logger.warning(f"Failed to compute loss for pair: {e}")
                continue

        if not hybrid_losses:
            logger.error("Failed to compute any hybrid losses")
            return {'hybrid_loss': None}

        # Average across all samples
        avg_hybrid_loss = np.mean(hybrid_losses)

        logger.info(f"Hybrid OCR Loss: {avg_hybrid_loss:.4f} (lower is better)")

        return {
            'hybrid_loss': avg_hybrid_loss,
            'num_samples': len(hybrid_losses)
        }

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1}/{self.num_epochs}"
        )

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.model.device) for k, v in batch.items() if v is not None}

            # Forward pass
            outputs = self.model(
                pixel_values=batch['pixel_values'],
                image_grid_thw=batch.get('image_grid_thw'),
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            # Base task loss (cross-entropy)
            task_loss = outputs['loss']

            # Hierarchical PEFT: Add KL divergence and L2-SP regularization
            if self.hierarchical_mode:
                # KL divergence to base model (bootstrap only)
                kl_loss = self._compute_kl_divergence_loss(batch, outputs['logits'])

                # L2-SP regularization (user adapter weights)
                l2sp_loss = self._compute_l2sp_loss()

                # Total loss
                loss = task_loss + self.kl_weight * kl_loss + self.l2sp_weight * l2sp_loss

                # Store components for logging
                task_loss_value = task_loss.item()
                kl_loss_value = kl_loss.item()
                l2sp_loss_value = l2sp_loss.item()
            else:
                loss = task_loss
                task_loss_value = task_loss.item()
                kl_loss_value = 0.0
                l2sp_loss_value = 0.0

            # Scale for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            total_loss += loss.item()
            num_batches += 1

            # Update weights every gradient_accumulation_steps
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Compute gradient norm before clipping
                grad_norm = self._compute_gradient_norm()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if self.global_step % self.log_every == 0:
                    avg_loss = total_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    perplexity = self._compute_perplexity(avg_loss)

                    # Log basic metrics
                    self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                    self.writer.add_scalar('train/learning_rate', lr, self.global_step)
                    self.writer.add_scalar('train/perplexity', perplexity, self.global_step)
                    self.writer.add_scalar('train/grad_norm', grad_norm, self.global_step)

                    # Log hierarchical PEFT metrics
                    if self.hierarchical_mode:
                        self.writer.add_scalar('train/task_loss', task_loss_value, self.global_step)
                        self.writer.add_scalar('train/kl_loss', kl_loss_value, self.global_step)
                        self.writer.add_scalar('train/l2sp_loss', l2sp_loss_value, self.global_step)

                    # Log parameter statistics (every 100 steps to reduce overhead)
                    if self.global_step % 100 == 0:
                        param_stats = self._log_parameter_stats()
                        for name, value in param_stats.items():
                            self.writer.add_scalar(name, value, self.global_step)

                    logger.info(
                        f"Step {self.global_step} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"PPL: {perplexity:.2f} | "
                        f"GradNorm: {grad_norm:.4f} | "
                        f"LR: {lr:.2e}"
                    )

                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'ppl': f'{perplexity:.2f}',
                        'grad': f'{grad_norm:.3f}',
                        'lr': f'{lr:.2e}'
                    })

                    # Reset accumulation
                    total_loss = 0
                    num_batches = 0

                # Evaluation
                if self.global_step % self.eval_every == 0:
                    self.evaluate()
                    self.model.train()

                # Save checkpoint
                if self.global_step % self.save_every == 0:
                    self.save_checkpoint(f'checkpoint-{self.global_step}')

        self.epoch += 1

    def evaluate(self):
        """Run evaluation on validation set with comprehensive metrics."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        logger.info("Running evaluation...")

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.model.device) for k, v in batch.items() if v is not None}

                # Forward pass
                outputs = self.model(
                    pixel_values=batch['pixel_values'],
                    image_grid_thw=batch.get('image_grid_thw'),
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )

                total_loss += outputs['loss'].item()
                num_batches += 1

        avg_val_loss = total_loss / num_batches
        val_perplexity = self._compute_perplexity(avg_val_loss)

        # Generate sample predictions for qualitative analysis
        logger.info("Generating sample predictions...")
        sample_predictions = self._generate_sample_predictions(num_samples=5)

        # Compute hybrid OCR loss on samples
        if sample_predictions:
            logger.info(f"Generated {len(sample_predictions)} sample predictions")
            predictions = [s['prediction'] for s in sample_predictions]
            ground_truths = [s['ground_truth'] for s in sample_predictions]

            # Log first sample for debugging
            if sample_predictions:
                logger.info(f"Sample 1 GT: {ground_truths[0][:50]}...")
                logger.info(f"Sample 1 Pred: {predictions[0][:50]}...")

            ocr_metrics = self._compute_hybrid_ocr_loss(predictions, ground_truths)

            # Log sample predictions as text
            samples_text = "\n\n".join([
                f"Sample {i+1}:\n"
                f"Ground Truth: {s['ground_truth']}\n"
                f"Prediction:   {s['prediction']}"
                for i, s in enumerate(sample_predictions)
            ])
            self.writer.add_text('val/sample_predictions', samples_text, self.global_step)

            # Log hybrid OCR loss (only if not None)
            if ocr_metrics['hybrid_loss'] is not None:
                self.writer.add_scalar('val/hybrid_ocr_loss', ocr_metrics['hybrid_loss'], self.global_step)
                logger.info(f"Validation Hybrid OCR Loss: {ocr_metrics['hybrid_loss']:.4f}")
            else:
                logger.error("Failed to compute hybrid OCR loss - predictions may be empty")
        else:
            ocr_metrics = {'hybrid_loss': None}
            logger.error("No sample predictions generated - check model.generate() method")

        # Log basic metrics
        self.writer.add_scalar('val/loss', avg_val_loss, self.global_step)
        self.writer.add_scalar('val/perplexity', val_perplexity, self.global_step)

        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"Validation Perplexity: {val_perplexity:.2f}")

        # Save best model based on hybrid OCR loss (if available) or validation loss
        if ocr_metrics['hybrid_loss'] is not None and ocr_metrics['hybrid_loss'] < self.best_hybrid_loss:
            self.best_hybrid_loss = ocr_metrics['hybrid_loss']
            self.save_checkpoint('best_model_hybrid')
            logger.info(f"New best model saved (hybrid_loss: {ocr_metrics['hybrid_loss']:.4f})")

        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.save_checkpoint('best_model_loss')
            logger.info(f"New best model saved (val_loss: {avg_val_loss:.4f})")

        return avg_val_loss

    def train(self):
        """Full training loop."""
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info("=" * 60)

        try:
            for epoch in range(self.num_epochs):
                logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
                self.train_epoch()

                # Evaluate at end of epoch
                self.evaluate()

            # Save final model
            self.save_checkpoint('final_model')

            logger.info("\n" + "=" * 60)
            logger.info("Training completed")
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
            logger.info(f"Checkpoints saved to: {self.output_dir}")
            logger.info("=" * 60)

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            self.save_checkpoint('interrupted')
            logger.info(f"Checkpoint saved to: {self.output_dir}/interrupted")

        finally:
            self.writer.close()

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter
        self.model.save_adapter(str(checkpoint_dir))

        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }

        torch.save(state, checkpoint_dir / 'training_state.pt')

        logger.info(f"Checkpoint saved: {checkpoint_dir}")


def main():
    """CLI for training."""
    parser = argparse.ArgumentParser(
        description="Train DisgraPhi vision-language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Bootstrap training on IAM dataset
  python train.py --mode bootstrap --model smolvlm-256m --experiment-name iam-base

  # Few-shot personalization (5-20 samples)
  python train.py --mode personalize --model smolvlm-256m \\
    --experiment-name user-john --user-dir data/john \\
    --bootstrap ml/checkpoints/bootstrap/best_model_cer \\
    --user-rank 2 --augment --kl 0.5

  # Full personalization (50+ samples)
  python train.py --mode personalize --model smolvlm-256m \\
    --user-dir data/john --bootstrap ml/checkpoints/bootstrap/best_model_cer \\
    --user-rank 4 --augment --kl 0.3 --epochs 10
        """
    )

    # === Core Settings ===
    parser.add_argument('--mode', type=str, required=True, choices=['bootstrap', 'personalize'],
                        help='bootstrap: train on IAM dataset | personalize: user-specific fine-tuning')
    parser.add_argument('--model', dest='model_type', type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture')
    parser.add_argument('--experiment-name', type=str, default='default',
                        help='Name for experiment tracking (tensorboard)')
    parser.add_argument('--output-dir', type=str, default='ml/checkpoints',
                        help='Base output directory (mode/experiment-name will be appended)')

    # === Data ===
    data_group = parser.add_argument_group('data')
    data_group.add_argument('--data-dir', type=str, default='ml/data/raw/iam/processed',
                           help='IAM dataset path (bootstrap mode)')
    data_group.add_argument('--sample-ratio', type=float, default=1.0,
                           help='Fraction of IAM dataset to use (0-1, default: 1.0). Use 0.5 for 50%% sampling')
    data_group.add_argument('--user-dir', type=str,
                           help='User data directory with manifest.json (personalize mode, required)')
    data_group.add_argument('--augment', action='store_true',
                           help='Enable handwriting augmentation (recommended for personalize)')
    data_group.add_argument('--augment-strength', type=float, default=0.7,
                           help='Augmentation intensity 0-1 (default: 0.7)')

    # === Model Architecture ===
    model_group = parser.add_argument_group('model architecture')
    model_group.add_argument('--lora-r', type=int, default=8,
                            help='LoRA rank for bootstrap (default: 8)')
    model_group.add_argument('--user-rank', type=int, default=2,
                            help='LoRA rank for user adapter in personalize mode (default: 2)')
    model_group.add_argument('--quant', type=str, choices=['4bit', '8bit', 'none'], default='4bit',
                            help='Quantization mode (default: 4bit)')

    # === Personalization (hierarchical PEFT) ===
    hier_group = parser.add_argument_group('personalization')
    hier_group.add_argument('--bootstrap', dest='bootstrap_adapter', type=str,
                           help='Path to bootstrap adapter checkpoint (personalize mode, required)')
    hier_group.add_argument('--kl', dest='kl_weight', type=float, default=0.5,
                           help='KL divergence weight for hierarchical training (default: 0.5)')
    hier_group.add_argument('--l2sp', dest='l2sp_weight', type=float, default=1e-4,
                           help='L2-SP regularization weight (default: 1e-4)')

    # === Training Hyperparameters ===
    train_group = parser.add_argument_group('training')
    train_group.add_argument('--lr', dest='learning_rate', type=float, default=None,
                            help='Learning rate (default: 1e-5 for bootstrap, 5e-4 for personalize)')
    train_group.add_argument('--batch-size', type=int, default=2,
                            help='Batch size per device (default: 2)')
    train_group.add_argument('--grad-accum', dest='gradient_accumulation_steps', type=int, default=4,
                            help='Gradient accumulation steps (default: 4)')
    train_group.add_argument('--epochs', dest='num_epochs', type=int, default=None,
                            help='Training epochs (default: 3 for bootstrap, auto for personalize)')
    train_group.add_argument('--eval-every', type=int, default=100,
                            help='Evaluation interval in steps (default: 100)')
    train_group.add_argument('--save-every', type=int, default=500,
                            help='Checkpoint save interval in steps (default: 500)')

    args = parser.parse_args()

    # === Smart Defaults Based on Mode ===
    if args.mode == 'bootstrap':
        args.learning_rate = args.learning_rate or 1e-5
        args.num_epochs = args.num_epochs or 3
        args.lora_alpha = args.lora_r * 2  # Standard scaling
        hierarchical_mode = False
    else:  # personalize
        if not args.user_dir:
            parser.error("--user-dir is required for personalize mode")
        if not args.bootstrap_adapter:
            parser.error("--bootstrap is required for personalize mode")

        args.learning_rate = args.learning_rate or 5e-4
        # Auto-set epochs based on dataset size if not specified
        if args.num_epochs is None:
            from pathlib import Path
            import json
            manifest_path = Path(args.user_dir) / 'manifest.json'
            if manifest_path.exists():
                with open(manifest_path) as f:
                    n_samples = json.load(f).get('totalSamples', 20)
                # More epochs for fewer samples
                args.num_epochs = max(10, min(100, 200 // n_samples))
            else:
                args.num_epochs = 20

        args.lora_r = args.user_rank  # Use user-rank for personalization
        args.lora_alpha = args.lora_r * 2
        hierarchical_mode = True

    # Quantization flags
    args.load_in_4bit = (args.quant == '4bit')
    args.load_in_8bit = (args.quant == '8bit')

    # Build full output path
    args.output_dir = f"{args.output_dir}/{args.mode}_{args.model_type}/{args.experiment_name}"

    # Print configuration summary
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {args.model_type}")
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Output: {args.output_dir}")
    if args.mode == 'personalize':
        logger.info(f"Bootstrap: {args.bootstrap_adapter}")
        logger.info(f"User data: {args.user_dir}")
        logger.info(f"User LoRA rank: {args.user_rank}")
        logger.info(f"Augmentation: {'enabled' if args.augment else 'disabled'}")
        logger.info(f"KL weight: {args.kl_weight}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info("=" * 60)

    # Create model using unified provider system
    logger.info(f"Creating model: {args.model_type}")
    model = create_model(
        model_type=args.model_type,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bootstrap_adapter_path=args.bootstrap_adapter if args.mode == 'personalize' else None
    )

    # For hierarchical mode, create base model (bootstrap only) for KL divergence
    base_model_for_kl = None
    if hierarchical_mode:
        logger.info("Creating base model for KL divergence...")
        base_model_for_kl = create_model(
            model_type=args.model_type,
            lora_r=8,  # Use bootstrap rank
            lora_alpha=16,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bootstrap_adapter_path=args.bootstrap_adapter
        )
        base_model_for_kl.eval()  # Freeze for KL computation

    # Load datasets
    if args.mode == 'bootstrap':
        logger.info("Loading IAM datasets...")
        if args.sample_ratio < 1.0:
            logger.info(f"Using {args.sample_ratio*100:.0f}% of IAM dataset for faster iteration")
        train_dataset = IAMDataset(
            data_dir=args.data_dir,
            split='train',
            sample_ratio=args.sample_ratio
        )
        val_dataset = IAMDataset(
            data_dir=args.data_dir,
            split='val',
            sample_ratio=args.sample_ratio
        )
    else:  # personalize
        logger.info(f"Loading user dataset from {args.user_dir}...")
        manifest_path = Path(args.user_dir) / 'manifest.json'

        if manifest_path.exists():
            logger.info("Using ManifestDataset")
            train_dataset = ManifestDataset(
                data_dir=args.user_dir,
                split=0.8,
                split_type='train',
                augment=args.augment,
                augment_strength=args.augment_strength
            )
            val_dataset = ManifestDataset(
                data_dir=args.user_dir,
                split=0.8,
                split_type='val',
                augment=False  # No augmentation on validation
            )
        else:
            logger.error("manifest.json not found - use new dataset format")
            return

    # Create trainer
    trainer = DisgraPhiTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        warmup_steps=0,  # Auto-compute
        eval_every=args.eval_every,
        save_every=args.save_every,
        hierarchical_mode=hierarchical_mode,
        kl_weight=args.kl_weight if hierarchical_mode else 0.0,
        l2sp_weight=args.l2sp_weight if hierarchical_mode else 0.0,
        base_model_for_kl=base_model_for_kl
    )

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
