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
from jiwer import cer, wer
from alveslib import get_logger

from ml.models.providers import create_model, MODEL_REGISTRY, BaseVisionLanguageModel
from ml.data.datasets import IAMDataset, PersonalizationDataset


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
        max_grad_norm: float = 1.0
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
        self.best_val_cer = float('inf')

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

        Processes images and text through the model's processor.
        Supports different model architectures:
        - Chat-based: Qwen3-VL, SmolVLM (use chat templates)
        - Prompt-based: Florence-2 (use task prompts)
        """
        images = [item['image'] for item in batch]
        texts = [item['text'] for item in batch]

        # Detect model type by checking if processor has a valid chat template
        # Florence-2 has the method but raises an error, so we check the attribute directly
        has_chat_template = (
            hasattr(self.model.processor, 'apply_chat_template') and
            hasattr(self.model.processor, 'chat_template') and
            self.model.processor.chat_template is not None
        )

        if has_chat_template:
            # Chat-based models (Qwen, SmolVLM)
            messages_batch = []
            for text in texts:
                messages_batch.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "Transcribe this handwritten text."}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": text}
                        ]
                    }
                ])

            texts_formatted = [
                self.model.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) if self.model.processor is not None else None
                for msg in messages_batch
            ]
        else: # TODO: If adding new model make this elsif
            # Prompt-based models (Florence-2)
            # Florence-2 format: <TASK_PROMPT>text</s>
            # For OCR task, we use <OCR> prompt
            texts_formatted = [f"<OCR>" for text in texts]

        # Process without truncation to preserve image tokens
        inputs = self.model.processor(
            text=texts_formatted,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False
        )

        # Handle long sequences
        max_seq_len = 1024  # Increase from 512 to accommodate image tokens
        if inputs['input_ids'].shape[1] > max_seq_len:
            inputs['input_ids'] = inputs['input_ids'][:, :max_seq_len]
            inputs['attention_mask'] = inputs['attention_mask'][:, :max_seq_len]

        # Create labels (shift input_ids by 1 for causal LM)
        labels = inputs['input_ids'].clone()
        labels[labels == self.model.tokenizer.pad_token_id] = -100

        return {
            'pixel_values': inputs['pixel_values'],
            'image_grid_thw': inputs.get('image_grid_thw'),  # Required by Qwen2-VL, optional for others
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels
        }

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

    def _compute_cer_wer(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """Compute Character Error Rate and Word Error Rate."""
        logger.info(f"Computing CER/WER for {len(predictions)} predictions")

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
            return {'cer': None, 'wer': None}  # Return None to indicate failure, not 0

        # Unzip into lists (not tuples) - jiwer expects list of strings
        preds, gts = zip(*valid_pairs)
        preds = list(preds)
        gts = list(gts)

        try:
            cer_score = cer(gts, preds) * 100  # Convert to percentage
            wer_score = wer(gts, preds) * 100
            logger.info(f"CER: {cer_score:.2f}%, WER: {wer_score:.2f}%")
        except Exception as e:
            logger.error(f"Error computing CER/WER: {e}")
            logger.error(f"Sample preds: {preds[:3]}")
            logger.error(f"Sample gts: {gts[:3]}")
            cer_score = None
            wer_score = None

        return {
            'cer': cer_score,
            'wer': wer_score
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

            loss = outputs['loss'] / self.gradient_accumulation_steps

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

        # Compute CER/WER on samples
        if sample_predictions:
            logger.info(f"Generated {len(sample_predictions)} sample predictions")
            predictions = [s['prediction'] for s in sample_predictions]
            ground_truths = [s['ground_truth'] for s in sample_predictions]

            # Log first sample for debugging
            if sample_predictions:
                logger.info(f"Sample 1 GT: {ground_truths[0][:50]}...")
                logger.info(f"Sample 1 Pred: {predictions[0][:50]}...")

            error_metrics = self._compute_cer_wer(predictions, ground_truths)

            # Log sample predictions as text
            samples_text = "\n\n".join([
                f"Sample {i+1}:\n"
                f"Ground Truth: {s['ground_truth']}\n"
                f"Prediction:   {s['prediction']}"
                for i, s in enumerate(sample_predictions)
            ])
            self.writer.add_text('val/sample_predictions', samples_text, self.global_step)

            # Log error metrics (only if not None)
            if error_metrics['cer'] is not None:
                self.writer.add_scalar('val/cer', error_metrics['cer'], self.global_step)
                self.writer.add_scalar('val/wer', error_metrics['wer'], self.global_step)
                logger.info(f"Validation CER: {error_metrics['cer']:.2f}%")
                logger.info(f"Validation WER: {error_metrics['wer']:.2f}%")
            else:
                logger.error("Failed to compute CER/WER - predictions may be empty")
        else:
            error_metrics = {'cer': None, 'wer': None}
            logger.error("No sample predictions generated - check model.generate() method")

        # Log basic metrics
        self.writer.add_scalar('val/loss', avg_val_loss, self.global_step)
        self.writer.add_scalar('val/perplexity', val_perplexity, self.global_step)

        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"Validation Perplexity: {val_perplexity:.2f}")

        # Save best model based on CER (if available) or loss
        if error_metrics['cer'] is not None and error_metrics['cer'] < self.best_val_cer:
            self.best_val_cer = error_metrics['cer']
            self.save_checkpoint('best_model_cer')
            logger.info(f"New best model saved (val_cer: {error_metrics['cer']:.2f}%)")

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
        description="Train DisgraPhi vision-language models (unified)"
    )

    # Model selection
    parser.add_argument(
        '--model-type',
        type=str,
        choices=list(MODEL_REGISTRY.keys()),
        required=True,
        help='Model type to train (e.g., qwen3-vl-4b, smolvlm-256m)'
    )

    # Training mode
    parser.add_argument(
        '--mode',
        type=str,
        choices=['bootstrap', 'personalize'],
        default='bootstrap',
        help='Training mode: bootstrap (IAM) or personalize (user-specific)'
    )

    # Data paths
    parser.add_argument(
        '--data-dir',
        type=str,
        default='ml/data/raw/iam/processed',
        help='Path to processed IAM data (for bootstrap mode)'
    )
    parser.add_argument(
        '--user-dir',
        type=str,
        help='Path to user data directory (for personalize mode)'
    )
    parser.add_argument(
        '--bootstrap-adapter',
        type=str,
        help='Path to bootstrap adapter (for personalize mode)'
    )

    # Model config
    parser.add_argument(
        '--lora-r',
        type=int,
        default=8,
        help='LoRA rank'
    )
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=16,
        help='LoRA alpha'
    )
    parser.add_argument(
        '--load-in-4bit',
        action='store_true',
        help='Enable 4-bit quantization'
    )
    parser.add_argument(
        '--load-in-8bit',
        action='store_true',
        help='Enable 8-bit quantization'
    )

    # Training config
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for checkpoints and logs'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='default',
        help='Experiment name for tensorboard logging'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Batch size per device'
    )
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=4,
        help='Gradient accumulation steps'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=0,
        help='Number of warmup steps (0 = auto: 30%% of total steps)'
    )
    parser.add_argument(
        '--eval-every',
        type=int,
        default=100,
        help='Evaluate every N steps'
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=500,
        help='Save checkpoint every N steps'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mode == 'personalize':
        if not args.user_dir:
            parser.error("--user-dir required for personalize mode")
        if not args.bootstrap_adapter:
            parser.error("--bootstrap-adapter required for personalize mode")

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

    # Load datasets
    if args.mode == 'bootstrap':
        logger.info("Loading IAM datasets...")
        train_dataset = IAMDataset(
            data_dir=args.data_dir,
            split='train'
        )
        val_dataset = IAMDataset(
            data_dir=args.data_dir,
            split='val'
        )
    else:  # personalize
        logger.info(f"Loading user dataset from {args.user_dir}...")
        full_dataset = PersonalizationDataset(user_dir=args.user_dir)

        # Split into train/val (80/20)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size]
        )

    # Create trainer
    experiment_name = args.experiment_name or f"{args.model_type}_{args.mode}"
    trainer = DisgraPhiTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        experiment_name=experiment_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        save_every=args.save_every
    )

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
