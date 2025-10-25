#!/usr/bin/env python3
"""
Ray-based parallel training orchestrator for A100

Features:
- Parallel training with GPU memory awareness
- Automatic queueing and scheduling
- Smart packing (run small models together, large models solo)
- Retry logic with exponential backoff
- Progress tracking and logging
- Automatic HuggingFace uploads

Usage:
    python ml/ray_train.py --sample-ratio 0.5 --hf-org velocitatem --parallel 2
    python ml/ray_train.py --sample-ratio 1.0 --hf-org myorg --parallel 1  # Sequential

Install Ray:
    pip install ray[default]
"""

import argparse
import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import ray
from ray.util.queue import Queue
import torch

# Ray actors for training
@ray.remote(num_gpus=1)
class ModelTrainer:
    """Actor that trains a single model on one GPU."""

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        print(f"[GPU {gpu_id}] Trainer initialized")

    def train(
        self,
        model_name: str,
        quant: str,
        experiment: str,
        sample_ratio: float,
        epochs: int,
        data_dir: str,
        output_dir: str
    ) -> Dict:
        """Train a model and return results."""
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        start_time = datetime.now()
        print(f"[{experiment}] Starting training on GPU {self.gpu_id}")

        cmd = [
            sys.executable, "ml/models/train.py",
            "--mode", "bootstrap",
            "--model", model_name,
            "--experiment-name", experiment,
            "--data-dir", data_dir,
            "--sample-ratio", str(sample_ratio),
            "--quant", quant,
            "--epochs", str(epochs),
            "--eval-every", "250",
            "--save-every", "500",
            "--batch-size", "2",
            "--grad-accum", "4"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=14400  # 4 hours
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            success = result.returncode == 0

            if success:
                print(f"[{experiment}] ✓ Completed in {duration:.0f}s")
            else:
                print(f"[{experiment}] ✗ Failed with exit code {result.returncode}")
                print(f"STDERR: {result.stderr[-500:]}")  # Last 500 chars

            return {
                'experiment': experiment,
                'model': model_name,
                'success': success,
                'exit_code': result.returncode,
                'duration': duration,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'checkpoint_dir': f"{output_dir}/bootstrap_{model_name}/{experiment}"
            }

        except subprocess.TimeoutExpired:
            print(f"[{experiment}] ✗ Timeout after 4 hours")
            return {
                'experiment': experiment,
                'success': False,
                'exit_code': -1,
                'error': 'timeout'
            }
        except Exception as e:
            print(f"[{experiment}] ✗ Exception: {e}")
            return {
                'experiment': experiment,
                'success': False,
                'exit_code': -2,
                'error': str(e)
            }


@ray.remote
class HuggingFaceUploader:
    """Actor for uploading models to HuggingFace."""

    def __init__(self, hf_org: str):
        self.hf_org = hf_org
        self._verify_auth()

    def _verify_auth(self):
        """Verify HF authentication."""
        try:
            subprocess.run(['huggingface-cli', 'whoami'], check=True, capture_output=True)
            print(f"✓ Authenticated with HuggingFace as {self.hf_org}")
        except subprocess.CalledProcessError:
            print("✗ Not authenticated with HuggingFace. Run: huggingface-cli login")
            raise

    def upload_checkpoints(
        self,
        checkpoint_dir: str,
        model_name: str,
        experiment: str,
        max_retries: int = 3
    ) -> bool:
        """Upload all checkpoints for a model."""
        from huggingface_hub import HfApi, create_repo

        checkpoint_base = Path(checkpoint_dir)
        if not checkpoint_base.exists():
            print(f"[{experiment}] ✗ Checkpoint directory not found: {checkpoint_base}")
            return False

        repo_name = f"disgraphi-bootstrap-{experiment}"
        repo_id = f"{self.hf_org}/{repo_name}"

        # Find checkpoint types
        checkpoint_types = ["best_model_hybrid", "best_model_loss", "final_model"]

        for checkpoint_type in checkpoint_types:
            checkpoint_path = checkpoint_base / checkpoint_type
            if not checkpoint_path.exists():
                continue

            # Create README
            self._create_readme(checkpoint_path, model_name, experiment, checkpoint_type)

            # Upload with retries
            for attempt in range(max_retries):
                try:
                    print(f"[{experiment}] Uploading {checkpoint_type} (attempt {attempt + 1}/{max_retries})")

                    api = HfApi()
                    create_repo(repo_id, exist_ok=True, private=False)

                    api.upload_folder(
                        folder_path=str(checkpoint_path),
                        repo_id=repo_id,
                        path_in_repo=checkpoint_type,
                        commit_message=f"Upload {checkpoint_type} checkpoint"
                    )

                    print(f"[{experiment}] ✓ Uploaded {checkpoint_type}")
                    break

                except Exception as e:
                    print(f"[{experiment}] Upload attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(10 * (attempt + 1))
                    else:
                        print(f"[{experiment}] ✗ Upload failed after {max_retries} attempts")
                        return False

        print(f"[{experiment}] ✓ All checkpoints uploaded to https://huggingface.co/{repo_id}")
        return True

    def _create_readme(self, checkpoint_dir: Path, model_name: str, experiment: str, checkpoint_type: str):
        """Create README.md."""
        readme = f"""---
license: apache-2.0
tags:
- handwriting-recognition
- vision-language
- lora
- disgraphi
- bootstrap
base_model: {model_name}
---

# DisgraPhi Bootstrap Adapter - {experiment}

Bootstrap LoRA adapter for handwriting recognition trained on IAM dataset.

## Model Details

- **Base Model**: {model_name}
- **Training Stage**: Bootstrap (global handwriting adapter)
- **Checkpoint**: {checkpoint_type}
- **Architecture**: LoRA rank 8, alpha 16
- **Trained**: {datetime.now().isoformat()}

## Usage

```python
from ml.models.providers import create_model

model = create_model(
    model_type="{model_name}",
    lora_r=8,
    lora_alpha=16,
    bootstrap_adapter_path="path/to/adapter"
)

text = model.generate(image, prompt="Transcribe this handwritten text.")
```

## Personalization

```bash
python train.py --mode personalize \\
  --model {model_name} \\
  --bootstrap path/to/this/adapter \\
  --user-dir data/user \\
  --user-rank 2 --augment --kl 0.5
```
"""
        (checkpoint_dir / "README.md").write_text(readme)


class RayOrchestrator:
    """Orchestrate parallel training with Ray."""

    def __init__(
        self,
        sample_ratio: float,
        hf_org: str,
        max_parallel: int = 2,
        epochs: int = 3,
        data_dir: str = "ml/data/raw/iam/processed",
        output_dir: str = "ml/checkpoints"
    ):
        self.sample_ratio = sample_ratio
        self.hf_org = hf_org
        self.max_parallel = max_parallel
        self.epochs = epochs
        self.data_dir = data_dir
        self.output_dir = output_dir

        # Initialize Ray
        if not ray.is_initialized():
            ray.init(num_gpus=1)  # One GPU available

        print(f"Ray initialized: {ray.cluster_resources()}")

        # Create uploader actor
        self.uploader = HuggingFaceUploader.remote(hf_org)

        # Define training jobs
        self.jobs = [
            {"model": "smolvlm-500m", "quant": "4bit", "experiment": "smolvlm500m-4bit", "vram_gb": 8},
            {"model": "qwen3-vl-2b", "quant": "4bit", "experiment": "qwen3-2b-4bit", "vram_gb": 14},
            {"model": "qwen3-vl-2b", "quant": "none", "experiment": "qwen3-2b-fp16", "vram_gb": 15},
            {"model": "qwen3-vl-4b", "quant": "4bit", "experiment": "qwen3-4b-4bit", "vram_gb": 20},
        ]

    def run(self):
        """Run all training jobs with smart scheduling."""
        print("=" * 80)
        print(f"Ray Parallel Training")
        print(f"Max parallel jobs: {self.max_parallel}")
        print(f"Sample ratio: {self.sample_ratio}")
        print(f"Total jobs: {len(self.jobs)}")
        print("=" * 80)

        results = []

        if self.max_parallel == 1:
            # Sequential mode
            results = self._run_sequential()
        else:
            # Parallel mode with smart packing
            results = self._run_parallel()

        self._print_summary(results)

        return results

    def _run_sequential(self):
        """Run jobs one at a time."""
        results = []
        trainer = ModelTrainer.remote(gpu_id=0)

        for job in self.jobs:
            print(f"\n[{job['experiment']}] Starting...")

            # Train
            result_ref = trainer.train.remote(
                model_name=job['model'],
                quant=job['quant'],
                experiment=job['experiment'],
                sample_ratio=self.sample_ratio,
                epochs=self.epochs,
                data_dir=self.data_dir,
                output_dir=self.output_dir
            )

            result = ray.get(result_ref)
            results.append(result)

            # Upload if successful
            if result['success']:
                upload_ref = self.uploader.upload_checkpoints.remote(
                    checkpoint_dir=result['checkpoint_dir'],
                    model_name=job['model'],
                    experiment=job['experiment']
                )
                ray.get(upload_ref)  # Wait for upload

            # Clear GPU memory between jobs
            torch.cuda.empty_cache()
            time.sleep(5)

        return results

    def _run_parallel(self):
        """Run jobs in parallel with smart packing on single GPU."""
        results = []

        # Pack jobs into batches that fit in GPU memory (40GB A100 with 35GB safe margin)
        batches = self._pack_jobs_into_batches()

        for batch_idx, batch in enumerate(batches):
            print(f"\n{'='*80}")
            print(f"[BATCH {batch_idx + 1}/{len(batches)}] Running {len(batch)} jobs in parallel")
            print(f"Models: {', '.join([j['experiment'] for j in batch])}")
            print(f"Total VRAM: {sum(j['vram_gb'] for j in batch)}GB")
            print(f"{'='*80}")

            # Create trainers for all jobs in batch
            trainers = [ModelTrainer.remote(gpu_id=0) for _ in batch]

            # Submit all jobs in batch
            result_refs = []
            for idx, (trainer, job) in enumerate(zip(trainers, batch)):
                if idx > 0:
                    time.sleep(10)  # Stagger starts

                result_ref = trainer.train.remote(
                    model_name=job['model'],
                    quant=job['quant'],
                    experiment=job['experiment'],
                    sample_ratio=self.sample_ratio,
                    epochs=self.epochs,
                    data_dir=self.data_dir,
                    output_dir=self.output_dir
                )
                result_refs.append(result_ref)

            # Wait for all jobs in batch to complete
            batch_results = ray.get(result_refs)
            results.extend(batch_results)

            # Upload successful models
            upload_refs = []
            for result, job in zip(batch_results, batch):
                if result['success']:
                    upload_ref = self.uploader.upload_checkpoints.remote(
                        result['checkpoint_dir'],
                        job['model'],
                        job['experiment']
                    )
                    upload_refs.append(upload_ref)

            # Wait for uploads to complete
            if upload_refs:
                ray.get(upload_refs)

            # Clear GPU between batches
            torch.cuda.empty_cache()
            time.sleep(10)

        return results

    def _pack_jobs_into_batches(self):
        """Pack jobs into batches that fit in GPU memory using greedy bin packing."""
        batches = []
        gpu_capacity = 35  # A100 40GB with 5GB safety margin

        # Sort jobs by VRAM requirements (largest first for better packing)
        sorted_jobs = sorted(self.jobs, key=lambda j: j['vram_gb'], reverse=True)
        remaining_jobs = sorted_jobs.copy()

        while remaining_jobs:
            current_batch = []
            current_vram = 0
            jobs_to_remove = []

            # Try to pack up to max_parallel jobs that fit in memory
            for job in remaining_jobs:
                can_add = (
                    len(current_batch) < self.max_parallel and
                    current_vram + job['vram_gb'] <= gpu_capacity
                )

                if can_add:
                    current_batch.append(job)
                    current_vram += job['vram_gb']
                    jobs_to_remove.append(job)

                    # Stop if we've reached max_parallel
                    if len(current_batch) == self.max_parallel:
                        break

            # If we couldn't pack any jobs, force add the largest one
            if not current_batch and remaining_jobs:
                print(f"[WARNING] Job '{remaining_jobs[0]['experiment']}' requires {remaining_jobs[0]['vram_gb']}GB (exceeds {gpu_capacity}GB limit)")
                current_batch.append(remaining_jobs[0])
                jobs_to_remove.append(remaining_jobs[0])

            # Remove packed jobs from remaining
            for job in jobs_to_remove:
                remaining_jobs.remove(job)

            if current_batch:
                batches.append(current_batch)

        return batches

    def _print_summary(self, results):
        """Print training summary."""
        print("\n" + "=" * 80)
        print("Training Summary")
        print("=" * 80)

        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful

        print(f"Total jobs: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        if successful > 0:
            avg_duration = sum(r.get('duration', 0) for r in results if r['success']) / successful
            print(f"Average duration: {avg_duration:.0f}s ({avg_duration/60:.1f} min)")

        print(f"\nAll models uploaded to: https://huggingface.co/{self.hf_org}")

        if failed > 0:
            print("\nFailed jobs:")
            for r in results:
                if not r['success']:
                    print(f"  - {r['experiment']} (exit code: {r.get('exit_code', 'unknown')})")

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Ray-based parallel training orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 2 models in parallel
  python ml/ray_train.py --sample-ratio 0.5 --hf-org velocitatem --parallel 2

  # Sequential training
  python ml/ray_train.py --sample-ratio 1.0 --hf-org myorg --parallel 1

  # Quick test
  python ml/ray_train.py --sample-ratio 0.1 --hf-org test --epochs 1 --parallel 2
        """
    )

    parser.add_argument('--sample-ratio', type=float, default=0.5,
                        help='Fraction of IAM dataset (default: 0.5)')
    parser.add_argument('--hf-org', type=str, required=True,
                        help='HuggingFace organization/username')
    parser.add_argument('--parallel', type=int, default=2,
                        help='Max parallel jobs (default: 2)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Training epochs (default: 3)')

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = RayOrchestrator(
        sample_ratio=args.sample_ratio,
        hf_org=args.hf_org,
        max_parallel=args.parallel,
        epochs=args.epochs
    )

    # Run training
    results = orchestrator.run()

    # Exit with error if any job failed
    if any(not r['success'] for r in results):
        sys.exit(1)


if __name__ == '__main__':
    main()
