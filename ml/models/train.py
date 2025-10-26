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
import re
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
from alveslib import get_logger

from ml.models.providers import create_model, MODEL_REGISTRY, BaseVisionLanguageModel
from ml.data.datasets import IAMDataset, PersonalizationDataset, ManifestDataset
from ml.models.eval import model_agnostic_loss, compute_ocr_metrics


logger = get_logger("ml-trainloop")


DEFAULT_SEED = 2025
EXPERIMENT_NAME_PATTERN = re.compile(r'^[a-z0-9]+(?:-[a-z0-9]+)?(?:\.[a-z0-9]+(?:-[a-z0-9]+)?){3}$')


def _slugify(value: str) -> str:
    """Convert an arbitrary string into a lowercase slug suitable for paths."""

    value = value.lower()
    value = re.sub(r'[^a-z0-9\-]+', '-', value)
    return value.strip('-') or 'unnamed'


def _resolve_seed(cli_seed: Optional[int] = None) -> int:
    """Derive the global random seed from CLI, environment, or defaults."""

    if cli_seed is not None:
        return cli_seed

    for env_key in ("DISGRAPHI_SEED", "SEED"):
        env_val = os.getenv(env_key)
        if env_val is not None:
            try:
                return int(env_val)
            except ValueError:
                logger.warning("Ignoring non-integer value for %s: %s", env_key, env_val)

    return DEFAULT_SEED


def _set_global_seed(seed: int) -> None:
    """Apply deterministic seeding across libraries."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Using global seed: %d", seed)


def _validate_experiment_name(name: str) -> Tuple[str, str, str, str]:
    """Ensure experiment naming follows task.model.variant.dataset hygiene."""

    if not EXPERIMENT_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "experiment-name must follow 'task.model.variant.dataset' using lowercase "
            "alphanumerics or dashes (e.g., ocr-bootstrap.smolvlm.256m.iam)"
        )

    task, model_arch, variant, dataset = name.split('.')
    return task, model_arch, variant, dataset


def _infer_experiment_name(args) -> Tuple[str, str, str, str, str]:
    """Return validated experiment name and its components."""

    if args.experiment_name:
        experiment = args.experiment_name
    else:
        task = _slugify(f"ocr-{args.mode}")
        model_parts = args.model_type.split('-', 1)
        model_arch = _slugify(model_parts[0])
        variant = _slugify(model_parts[1]) if len(model_parts) > 1 else 'base'
        if args.mode == 'bootstrap':
            data_path = Path(args.data_dir)
            dataset_name = data_path.name or 'iam'
            if dataset_name == 'processed':
                dataset_name = data_path.parent.name or dataset_name
            dataset = _slugify(dataset_name or 'iam')
        else:
            dataset = _slugify(Path(args.user_dir).name)
        experiment = '.'.join([task, model_arch, variant, dataset])

    task, model_arch, variant, dataset = _validate_experiment_name(experiment)
    return experiment, task, model_arch, variant, dataset

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
        test_dataset=None,
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
        self.test_dataset = test_dataset
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

        self.test_loader = None
        if test_dataset is not None:
            self.test_loader = DataLoader(
                test_dataset,
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
        if self.test_dataset is not None:
            logger.info(f"  Test samples: {len(self.test_dataset)}")
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
        processed = self.model.prepare_training_batch(images, texts)
        processed['ground_truth_texts'] = texts
        return processed

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move tensors in the batch to the model device, keep metadata intact."""

        result: Dict[str, Any] = {}
        for key, value in batch.items():
            if value is None:
                continue

            if hasattr(value, 'to'):
                result[key] = value.to(self.model.device)
            else:
                result[key] = value

        return result

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

    def _generate_predictions(
        self,
        dataset,
        limit: Optional[int] = None,
        description: Optional[str] = None
    ) -> Tuple[List[str], List[str]]:
        """Generate predictions for a dataset (optionally limited)."""

        total = len(dataset)
        max_items = total if limit is None else min(limit, total)
        if max_items == 0:
            return [], []

        iterator = range(max_items)
        if description:
            iterator = tqdm(iterator, desc=description)

        predictions: List[str] = []
        ground_truths: List[str] = []

        self.model.eval()
        with torch.no_grad():
            for idx in iterator:
                sample = dataset[idx]
                image = sample['image']
                ground_truth = sample['text']

                try:
                    image_input = image.convert('RGB') if hasattr(image, 'convert') else image
                    prediction = self.model.generate(
                        pixel_values=image_input,
                        prompt="Transcribe this handwritten text.",
                        max_new_tokens=128,
                        temperature=0.0
                    )

                    if isinstance(prediction, tuple):
                        prediction_text = prediction[0]
                    else:
                        prediction_text = prediction

                    predictions.append(str(prediction_text))
                    ground_truths.append(str(ground_truth))

                except Exception as exc:  # pragma: no cover - generation depends on model
                    logger.warning("Failed to generate prediction for sample %s: %s", idx, exc)
                    continue

        return predictions, ground_truths

    def _compute_full_metrics(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> Optional[Dict[str, float]]:
        """Compute averaged OCR metrics across all prediction pairs."""

        valid_pairs = [
            (pred.strip(), gt.strip())
            for pred, gt in zip(predictions, ground_truths)
            if gt.strip()
        ]

        if not valid_pairs:
            return None

        totals = {
            'cer': 0.0,
            'wer': 0.0,
            'ned': 0.0,
            'accuracy': 0.0,
            'hybrid_loss': 0.0
        }

        for pred, gt in valid_pairs:
            metrics = compute_ocr_metrics(gt, pred)
            totals['cer'] += metrics['cer']
            totals['wer'] += metrics['wer']
            totals['ned'] += metrics['ned']
            totals['accuracy'] += metrics['accuracy']
            totals['hybrid_loss'] += model_agnostic_loss(gt, pred)

        count = len(valid_pairs)
        averaged = {key: value / count for key, value in totals.items()}
        averaged['num_samples'] = count
        return averaged

    def _benchmark_dataset(self, dataset, split_name: str) -> Optional[Dict[str, float]]:
        """Run full benchmark over a dataset and return averaged metrics."""

        logger.info("Running %s hybrid OCR benchmark on %d samples", split_name, len(dataset))
        predictions, ground_truths = self._generate_predictions(
            dataset,
            limit=None,
            description=f"{split_name} benchmark"
        )
        metrics = self._compute_full_metrics(predictions, ground_truths)
        if metrics is None:
            logger.error("No valid prediction pairs generated for %s benchmark", split_name)
            return None

        logger.info(
            "%s metrics | hybrid=%.4f cer=%.4f wer=%.4f ned=%.4f accuracy=%.4f (n=%d)",
            split_name,
            metrics['hybrid_loss'],
            metrics['cer'],
            metrics['wer'],
            metrics['ned'],
            metrics['accuracy'],
            metrics['num_samples']
        )
        return metrics

    def _generate_sample_predictions(
        self,
        num_samples: int = 3
    ) -> List[Dict[str, str]]:
        """Generate sample predictions for qualitative evaluation."""
        predictions, ground_truths = self._generate_predictions(
            self.val_dataset,
            limit=num_samples
        )

        return [
            {'ground_truth': gt, 'prediction': pred}
            for pred, gt in zip(predictions, ground_truths)
        ]

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
            # Move batch to device while preserving metadata
            batch = self._move_batch_to_device(batch)

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
                    self.evaluate(compute_benchmark=False)
                    self.model.train()

                # Save checkpoint
                if self.global_step % self.save_every == 0:
                    self.save_checkpoint(f'checkpoint-{self.global_step}')

        self.epoch += 1

    def evaluate(
        self,
        loader=None,
        split_name: str = 'val',
        compute_benchmark: bool = True
    ) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        """Run evaluation on a loader and optionally compute the hybrid benchmark."""

        loader = loader or self.val_loader
        if loader is None:
            logger.warning("No %s loader available for evaluation", split_name)
            return None, None

        dataset = self.val_dataset if loader is self.val_loader else self.test_dataset
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        logger.info("Running %s evaluation...", split_name)

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating ({split_name})"):
                batch = self._move_batch_to_device(batch)
                outputs = self.model(
                    pixel_values=batch['pixel_values'],
                    image_grid_thw=batch.get('image_grid_thw'),
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )

                total_loss += outputs['loss'].item()
                num_batches += 1

        if num_batches == 0:
            logger.warning("No batches processed during %s evaluation", split_name)
            return None, None

        avg_loss = total_loss / num_batches
        perplexity = self._compute_perplexity(avg_loss)

        self.writer.add_scalar(f'{split_name}/loss', avg_loss, self.global_step)
        self.writer.add_scalar(f'{split_name}/perplexity', perplexity, self.global_step)

        logger.info("%s Loss: %.4f", split_name.capitalize(), avg_loss)
        logger.info("%s Perplexity: %.2f", split_name.capitalize(), perplexity)

        benchmark_metrics: Optional[Dict[str, float]] = None

        if compute_benchmark and dataset is not None:
            benchmark_metrics = self._benchmark_dataset(dataset, split_name)
            if benchmark_metrics:
                self.writer.add_scalar(f'{split_name}/hybrid_ocr_loss', benchmark_metrics['hybrid_loss'], self.global_step)
                self.writer.add_scalar(f'{split_name}/cer', benchmark_metrics['cer'], self.global_step)
                self.writer.add_scalar(f'{split_name}/wer', benchmark_metrics['wer'], self.global_step)
                self.writer.add_scalar(f'{split_name}/ned', benchmark_metrics['ned'], self.global_step)
                self.writer.add_scalar(f'{split_name}/accuracy', benchmark_metrics['accuracy'], self.global_step)

                samples = self._generate_sample_predictions(num_samples=5)
                if samples:
                    samples_text = "\n\n".join([
                        f"Sample {i+1}:\nGround Truth: {s['ground_truth']}\nPrediction:   {s['prediction']}"
                        for i, s in enumerate(samples)
                    ])
                    self.writer.add_text(f'{split_name}/sample_predictions', samples_text, self.global_step)

        if split_name == 'val' and benchmark_metrics:
            if benchmark_metrics['hybrid_loss'] < self.best_hybrid_loss:
                self.best_hybrid_loss = benchmark_metrics['hybrid_loss']
                self.save_checkpoint('best_model_hybrid')
                logger.info(
                    "New best model saved (hybrid_loss: %.4f)",
                    benchmark_metrics['hybrid_loss']
                )

        if split_name == 'val' and avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint('best_model_loss')
            logger.info(f"New best model saved (val_loss: {avg_loss:.4f})")

        return avg_loss, benchmark_metrics

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
                self.evaluate(split_name='val', compute_benchmark=True)

            if self.test_loader is not None:
                self.evaluate(
                    loader=self.test_loader,
                    split_name='test',
                    compute_benchmark=True
                )

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
  python train.py --mode bootstrap --model smolvlm-256m \\
    --experiment-name ocr-bootstrap.smolvlm.256m.iam

  # Few-shot personalization (5-20 samples)
  python train.py --mode personalize --model smolvlm-256m \\
    --user-dir data/john --bootstrap ml/checkpoints/bootstrap/... \\
    --experiment-name ocr-personalize.smolvlm.256m.john \\
    --user-rank 2 --augment --kl 0.5

  # Full personalization (50+ samples)
  python train.py --mode personalize --model smolvlm-256m \\
    --user-dir data/john --bootstrap ml/checkpoints/bootstrap/... \\
    --experiment-name ocr-personalize.smolvlm.256m.john \\
    --user-rank 4 --augment --kl 0.3 --epochs 10
        """
    )

    # === Core Settings ===
    parser.add_argument('--mode', type=str, required=True, choices=['bootstrap', 'personalize'],
                        help='bootstrap: train on IAM dataset | personalize: user-specific fine-tuning')
    parser.add_argument('--model', dest='model_type', type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help="Explicit experiment name following 'task.model.variant.dataset'")
    parser.add_argument('--output-dir', type=str, default='ml/checkpoints',
                        help='Base output directory (task/model/variant/dataset folders appended)')

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
    train_group.add_argument('--seed', type=int, default=None,
                             help='Override random seed (env DISGRAPHI_SEED/SEED otherwise, default 2025)')

    args = parser.parse_args()

    seed = _resolve_seed(args.seed)
    _set_global_seed(seed)

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

    experiment_name, task_token, model_arch_token, variant_token, dataset_token = _infer_experiment_name(args)
    args.experiment_name = experiment_name

    # Build full output path following naming hygiene
    output_path = Path(args.output_dir) / task_token / model_arch_token / variant_token / dataset_token
    args.output_dir = str(output_path)

    # Print configuration summary
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {args.model_type}")
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Seed: {seed}")
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
    test_dataset = None
    if args.mode == 'bootstrap':
        logger.info("Loading IAM datasets...")
        if args.sample_ratio < 1.0:
            logger.info(f"Using {args.sample_ratio*100:.0f}% of IAM dataset for faster iteration")
        train_dataset = IAMDataset(
            data_dir=args.data_dir,
            split='train',
            sample_ratio=args.sample_ratio,
            seed=seed
        )
        val_dataset = IAMDataset(
            data_dir=args.data_dir,
            split='val',
            sample_ratio=args.sample_ratio,
            seed=seed
        )
        test_dataset = IAMDataset(
            data_dir=args.data_dir,
            split='test',
            sample_ratio=args.sample_ratio,
            seed=seed
        )
    else:  # personalize
        logger.info(f"Loading user dataset from {args.user_dir}...")
        manifest_path = Path(args.user_dir) / 'manifest.json'

        if manifest_path.exists():
            logger.info("Using ManifestDataset")
            split_ratios = (0.7, 0.2, 0.1)
            try:
                train_dataset = ManifestDataset(
                    data_dir=args.user_dir,
                    subset='train',
                    split_ratios=split_ratios,
                    seed=seed,
                    augment=args.augment,
                    augment_strength=args.augment_strength
                )
                val_dataset = ManifestDataset(
                    data_dir=args.user_dir,
                    subset='val',
                    split_ratios=split_ratios,
                    seed=seed,
                    augment=False
                )
                test_dataset = ManifestDataset(
                    data_dir=args.user_dir,
                    subset='test',
                    split_ratios=split_ratios,
                    seed=seed,
                    augment=False
                )
            except ValueError as exc:
                logger.warning(
                    "Falling back to 80/20 train/val split (no test set): %s",
                    exc
                )
                fallback_ratios = (0.8, 0.2, 0.0)
                train_dataset = ManifestDataset(
                    data_dir=args.user_dir,
                    subset='train',
                    split_ratios=fallback_ratios,
                    seed=seed,
                    augment=args.augment,
                    augment_strength=args.augment_strength
                )
                val_dataset = ManifestDataset(
                    data_dir=args.user_dir,
                    subset='val',
                    split_ratios=fallback_ratios,
                    seed=seed,
                    augment=False
                )
                test_dataset = None
        else:
            logger.error("manifest.json not found - use new dataset format")
            return

    # Create trainer
    trainer = DisgraPhiTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
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
