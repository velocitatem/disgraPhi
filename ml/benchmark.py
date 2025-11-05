"""
Comprehensive benchmarking framework for DisgraPhi.

Evaluates models on:
- OCR metrics (CER, WER, NED, accuracy)
- Inference speed (latency, throughput)
- Memory usage (VRAM during inference)
- Adapter size
- Per-writer performance analysis
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import torch
from tqdm import tqdm

from ml.models.providers import create_model
from ml.models.eval import compute_ocr_metrics
from ml.data.datasets import IAMDataset


@dataclass
class BenchmarkResult:
    """Results for a single model benchmark."""
    model_name: str
    model_provider: str
    quantization: str
    adapter_path: Optional[str]

    # OCR Metrics
    cer_mean: float
    cer_std: float
    wer_mean: float
    wer_std: float
    ned_mean: float
    ned_std: float
    accuracy: float

    # Performance
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    throughput_samples_per_sec: float

    # Memory
    vram_mb: float
    adapter_size_mb: Optional[float]

    # Dataset
    num_samples: int
    dataset_name: str

    # Per-writer stats (if available)
    cer_per_writer: Optional[Dict[str, float]] = None
    worst_writer: Optional[str] = None
    best_writer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelBenchmark:
    """Benchmark a single model on a test dataset."""

    def __init__(
        self,
        model_provider: str,
        adapter_path: Optional[str] = None,
        quantization: str = '4bit',
        device: str = 'cuda'
    ):
        self.model_provider = model_provider
        self.adapter_path = adapter_path
        self.quantization = quantization
        self.device = device

        print(f"Loading model: {model_provider} ({quantization})")
        if adapter_path:
            print(f"  Adapter: {adapter_path}")

        # Load model
        self.model = create_model(
            model_type=model_provider,
            load_in_4bit=(quantization == '4bit'),
            load_in_8bit=(quantization == '8bit'),
            bootstrap_adapter_path=adapter_path
        )

        # Warmup
        print("Warming up model...")
        self._warmup()

    def _warmup(self, num_warmup: int = 3):
        """Warmup model with dummy samples."""
        from PIL import Image
        dummy_img = Image.new('RGB', (200, 50), color='white')

        for _ in range(num_warmup):
            _ = self.model.generate(dummy_img, prompt="test")

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def benchmark_dataset(
        self,
        dataset: IAMDataset,
        max_samples: Optional[int] = None
    ) -> BenchmarkResult:
        """
        Benchmark model on dataset.

        Args:
            dataset: Test dataset
            max_samples: Maximum number of samples to evaluate (None = all)

        Returns:
            BenchmarkResult object
        """
        print(f"\nBenchmarking on {len(dataset)} samples...")
        if max_samples:
            dataset_size = min(max_samples, len(dataset))
        else:
            dataset_size = len(dataset)

        # Metrics storage
        cer_scores = []
        wer_scores = []
        ned_scores = []
        exact_matches = []
        latencies = []
        per_writer_cer = {}

        # Measure initial VRAM
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_vram = torch.cuda.memory_allocated() / 1024**2

        # Evaluate samples
        for idx in tqdm(range(dataset_size), desc="Evaluating"):
            sample = dataset[idx]
            image = sample['image']
            ground_truth = sample['text']

            # Time inference
            start_time = time.time()
            prediction = self.model.generate(image, prompt="Transcribe this handwritten text:")
            end_time = time.time()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            # Compute metrics
            metrics = compute_ocr_metrics(ground_truth, prediction)
            cer_scores.append(metrics['cer'])
            wer_scores.append(metrics['wer'])
            ned_scores.append(metrics['ned'])
            exact_matches.append(1.0 if ground_truth == prediction else 0.0)

            # Per-writer stats (if writer ID available)
            if 'writer_id' in sample:
                writer_id = sample['writer_id']
                if writer_id not in per_writer_cer:
                    per_writer_cer[writer_id] = []
                per_writer_cer[writer_id].append(metrics['cer'])

        # Compute aggregated metrics
        cer_mean = np.mean(cer_scores)
        cer_std = np.std(cer_scores)
        wer_mean = np.mean(wer_scores)
        wer_std = np.std(wer_scores)
        ned_mean = np.mean(ned_scores)
        ned_std = np.std(ned_scores)
        accuracy = np.mean(exact_matches)

        # Latency statistics
        latencies_sorted = np.sort(latencies)
        latency_mean = np.mean(latencies)
        latency_p50 = np.percentile(latencies, 50)
        latency_p95 = np.percentile(latencies, 95)
        latency_p99 = np.percentile(latencies, 99)
        throughput = 1000.0 / latency_mean  # samples per second

        # VRAM usage
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / 1024**2
            vram_mb = peak_vram
        else:
            vram_mb = 0.0

        # Adapter size
        adapter_size_mb = None
        if self.adapter_path:
            adapter_path = Path(self.adapter_path)
            if adapter_path.exists():
                total_size = sum(
                    f.stat().st_size for f in adapter_path.rglob('*') if f.is_file()
                )
                adapter_size_mb = total_size / 1024**2

        # Per-writer analysis
        per_writer_cer_mean = {
            writer: np.mean(cers) for writer, cers in per_writer_cer.items()
        }
        worst_writer = max(per_writer_cer_mean.items(), key=lambda x: x[1])[0] if per_writer_cer_mean else None
        best_writer = min(per_writer_cer_mean.items(), key=lambda x: x[1])[0] if per_writer_cer_mean else None

        return BenchmarkResult(
            model_name=self.model_provider,
            model_provider=self.model_provider,
            quantization=self.quantization,
            adapter_path=self.adapter_path,
            cer_mean=cer_mean,
            cer_std=cer_std,
            wer_mean=wer_mean,
            wer_std=wer_std,
            ned_mean=ned_mean,
            ned_std=ned_std,
            accuracy=accuracy,
            latency_ms_mean=latency_mean,
            latency_ms_p50=latency_p50,
            latency_ms_p95=latency_p95,
            latency_ms_p99=latency_p99,
            throughput_samples_per_sec=throughput,
            vram_mb=vram_mb,
            adapter_size_mb=adapter_size_mb,
            num_samples=dataset_size,
            dataset_name="IAM",
            cer_per_writer=per_writer_cer_mean,
            worst_writer=worst_writer,
            best_writer=best_writer
        )


def generate_leaderboard(results: List[BenchmarkResult], output_path: Path):
    """Generate markdown leaderboard from benchmark results."""
    lines = []
    lines.append("# DisgraPhi Model Leaderboard")
    lines.append("")
    lines.append("Benchmarked on IAM test set")
    lines.append("")

    # Sort by CER (lower is better)
    results_sorted = sorted(results, key=lambda r: r.cer_mean)

    # Accuracy table
    lines.append("## OCR Accuracy")
    lines.append("")
    lines.append("| Rank | Model | Quantization | CER↓ | WER↓ | NED↓ | Accuracy↑ | Adapter Size |")
    lines.append("|------|-------|--------------|------|------|------|-----------|--------------|")

    for rank, result in enumerate(results_sorted, 1):
        adapter_size = f"{result.adapter_size_mb:.1f}MB" if result.adapter_size_mb else "N/A"
        lines.append(
            f"| {rank} | {result.model_name} | {result.quantization} | "
            f"{result.cer_mean:.3f}±{result.cer_std:.3f} | "
            f"{result.wer_mean:.3f}±{result.wer_std:.3f} | "
            f"{result.ned_mean:.3f}±{result.ned_std:.3f} | "
            f"{result.accuracy:.3f} | {adapter_size} |"
        )

    # Performance table
    lines.append("")
    lines.append("## Inference Performance")
    lines.append("")
    lines.append("| Model | Latency (ms) | P95 | P99 | Throughput | VRAM |")
    lines.append("|-------|--------------|-----|-----|------------|------|")

    # Sort by latency
    results_by_speed = sorted(results, key=lambda r: r.latency_ms_mean)

    for result in results_by_speed:
        lines.append(
            f"| {result.model_name} | "
            f"{result.latency_ms_mean:.1f} | "
            f"{result.latency_ms_p95:.1f} | "
            f"{result.latency_ms_p99:.1f} | "
            f"{result.throughput_samples_per_sec:.2f}/s | "
            f"{result.vram_mb:.0f}MB |"
        )

    # Recommendations
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")

    best_accuracy = results_sorted[0]
    fastest = results_by_speed[0]
    smallest_vram = min(results, key=lambda r: r.vram_mb)

    lines.append(f"- **Best Accuracy**: {best_accuracy.model_name} (CER: {best_accuracy.cer_mean:.3f})")
    lines.append(f"- **Fastest**: {fastest.model_name} ({fastest.latency_ms_mean:.1f}ms latency)")
    lines.append(f"- **Most Efficient**: {smallest_vram.model_name} ({smallest_vram.vram_mb:.0f}MB VRAM)")

    lines.append("")
    lines.append("*Lower CER/WER/NED and higher Accuracy are better*")
    lines.append("*Lower latency and VRAM usage are better*")

    # Write to file
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"\n✓ Leaderboard saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark DisgraPhi models")
    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='Model providers to benchmark (e.g., deepseek-ocr smolvlm-256m)'
    )
    parser.add_argument(
        '--adapters',
        nargs='+',
        help='Adapter paths (same order as models)'
    )
    parser.add_argument(
        '--quantization',
        nargs='+',
        default=['4bit'],
        help='Quantization for each model (e.g., 4bit 8bit none)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./ml/data/raw/iam',
        help='Path to IAM dataset'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        help='Maximum samples to evaluate (for quick testing)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./ml/LEADERBOARD.md',
        help='Output leaderboard path'
    )
    parser.add_argument(
        '--json',
        type=str,
        help='Optional JSON output path'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("DisgraPhi Model Benchmark")
    print("=" * 80)

    # Load test dataset
    print(f"\nLoading IAM test dataset from {args.data_dir}...")
    test_dataset = IAMDataset(data_dir=args.data_dir, split='test')
    print(f"✓ Loaded {len(test_dataset)} test samples")

    # Prepare models
    models = args.models
    adapters = args.adapters if args.adapters else [None] * len(models)
    quantizations = args.quantization * len(models) if len(args.quantization) == 1 else args.quantization

    if len(adapters) != len(models):
        raise ValueError(f"Number of adapters ({len(adapters)}) must match number of models ({len(models)})")
    if len(quantizations) != len(models):
        raise ValueError(f"Number of quantizations ({len(quantizations)}) must match number of models ({len(models)})")

    # Benchmark each model
    results = []

    for model, adapter, quant in zip(models, adapters, quantizations):
        print(f"\n{'='*80}")
        print(f"Benchmarking: {model} ({quant})")
        if adapter:
            print(f"Adapter: {adapter}")
        print('='*80)

        try:
            benchmark = ModelBenchmark(
                model_provider=model,
                adapter_path=adapter,
                quantization=quant
            )

            result = benchmark.benchmark_dataset(
                dataset=test_dataset,
                max_samples=args.max_samples
            )

            results.append(result)

            # Print summary
            print(f"\nResults:")
            print(f"  CER: {result.cer_mean:.3f} ± {result.cer_std:.3f}")
            print(f"  WER: {result.wer_mean:.3f} ± {result.wer_std:.3f}")
            print(f"  Accuracy: {result.accuracy:.3f}")
            print(f"  Latency: {result.latency_ms_mean:.1f}ms (p95: {result.latency_ms_p95:.1f}ms)")
            print(f"  Throughput: {result.throughput_samples_per_sec:.2f} samples/sec")
            print(f"  VRAM: {result.vram_mb:.0f}MB")

            # Clear GPU memory between models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"✗ Failed to benchmark {model}: {e}")
            continue

    if not results:
        print("\n✗ No successful benchmarks")
        return

    # Generate leaderboard
    print(f"\n{'='*80}")
    print("Generating leaderboard...")
    print('='*80)

    output_path = Path(args.output)
    generate_leaderboard(results, output_path)

    # Save JSON if requested
    if args.json:
        json_path = Path(args.json)
        with open(json_path, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"✓ JSON results saved to: {json_path}")

    print(f"\n{'='*80}")
    print("Benchmark Complete!")
    print('='*80)


if __name__ == "__main__":
    main()
