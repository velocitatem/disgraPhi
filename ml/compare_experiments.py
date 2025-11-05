"""
Experiment comparison tool for DisgraPhi.

Reads TensorBoard logs and generates comparison tables for:
- Training metrics (loss, CER, WER, NED)
- Model specifications (size, quantization, LoRA config)
- Training speed and efficiency
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import re

try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


@dataclass
class ExperimentMetrics:
    """Metrics for a single experiment."""
    name: str
    model: str
    dataset: str
    precision: str

    # Final metrics
    final_train_loss: Optional[float] = None
    final_eval_loss: Optional[float] = None
    best_eval_loss: Optional[float] = None
    final_cer: Optional[float] = None
    final_wer: Optional[float] = None
    final_ned: Optional[float] = None
    final_accuracy: Optional[float] = None

    # Training info
    total_steps: Optional[int] = None
    total_epochs: Optional[int] = None
    training_time_hours: Optional[float] = None
    samples_per_second: Optional[float] = None

    # Model info
    model_size_params: Optional[int] = None
    trainable_params: Optional[int] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_experiment_name(name: str) -> Dict[str, str]:
    """
    Parse experiment name into components.

    Format: [trl.]task.model.size.dataset.precision
    Example: trl.bootstrap.deepseekocr.3b.iam100.4bit
    """
    parts = name.split('.')

    # Check if TRL prefix exists
    if parts[0] == 'trl':
        parts = parts[1:]

    if len(parts) >= 5:
        return {
            'task': parts[0],
            'model': parts[1],
            'size': parts[2],
            'dataset': parts[3],
            'precision': parts[4]
        }
    else:
        return {
            'task': 'unknown',
            'model': name,
            'size': 'unknown',
            'dataset': 'unknown',
            'precision': 'unknown'
        }


def load_tensorboard_metrics(log_dir: Path) -> Optional[ExperimentMetrics]:
    """
    Load metrics from TensorBoard event files.

    Args:
        log_dir: Path to experiment log directory

    Returns:
        ExperimentMetrics object or None if loading fails
    """
    if not TENSORBOARD_AVAILABLE:
        print("Warning: tensorboard not installed. Install with: pip install tensorboard")
        return None

    if not log_dir.exists():
        print(f"Warning: Log directory not found: {log_dir}")
        return None

    # Parse experiment name
    exp_name = log_dir.name
    parsed = parse_experiment_name(exp_name)

    metrics = ExperimentMetrics(
        name=exp_name,
        model=f"{parsed['model']}-{parsed['size']}",
        dataset=parsed['dataset'],
        precision=parsed['precision']
    )

    try:
        # Load TensorBoard events
        ea = event_accumulator.EventAccumulator(str(log_dir))
        ea.Reload()

        # Get available scalar tags
        tags = ea.Tags().get('scalars', [])

        # Extract final metrics
        for tag in tags:
            events = ea.Scalars(tag)
            if not events:
                continue

            final_value = events[-1].value
            final_step = events[-1].step

            if tag == 'train/loss':
                metrics.final_train_loss = final_value
                metrics.total_steps = final_step
            elif tag == 'eval/loss':
                metrics.final_eval_loss = final_value
                # Find best eval loss
                metrics.best_eval_loss = min(e.value for e in events)
            elif tag == 'eval/cer':
                metrics.final_cer = final_value
            elif tag == 'eval/wer':
                metrics.final_wer = final_value
            elif tag == 'eval/ned':
                metrics.final_ned = final_value
            elif tag == 'eval/accuracy':
                metrics.final_accuracy = final_value

        return metrics

    except Exception as e:
        print(f"Error loading metrics from {log_dir}: {e}")
        return None


def load_all_experiments(logs_dir: Path) -> List[ExperimentMetrics]:
    """
    Load metrics from all experiments in logs directory.

    Args:
        logs_dir: Path to logs directory containing experiment subdirectories

    Returns:
        List of ExperimentMetrics
    """
    experiments = []

    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir}")
        return experiments

    for exp_dir in logs_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        metrics = load_tensorboard_metrics(exp_dir)
        if metrics:
            experiments.append(metrics)

    return experiments


def generate_comparison_table(experiments: List[ExperimentMetrics]) -> str:
    """
    Generate markdown comparison table.

    Args:
        experiments: List of ExperimentMetrics

    Returns:
        Markdown table string
    """
    if not experiments:
        return "No experiments found."

    # Sort by best eval loss
    experiments_sorted = sorted(
        experiments,
        key=lambda e: e.best_eval_loss if e.best_eval_loss is not None else float('inf')
    )

    # Build table
    lines = []
    lines.append("# Experiment Comparison")
    lines.append("")
    lines.append("## Bootstrap Models")
    lines.append("")

    # Header
    lines.append("| Experiment | Model | Dataset | CER↓ | WER↓ | NED↓ | Accuracy↑ | Loss↓ | Precision |")
    lines.append("|------------|-------|---------|------|------|------|-----------|-------|-----------|")

    for exp in experiments_sorted:
        if 'bootstrap' in exp.name.lower():
            lines.append(
                f"| {exp.name[:40]} | {exp.model} | {exp.dataset} | "
                f"{exp.final_cer:.3f if exp.final_cer else 'N/A'} | "
                f"{exp.final_wer:.3f if exp.final_wer else 'N/A'} | "
                f"{exp.final_ned:.3f if exp.final_ned else 'N/A'} | "
                f"{exp.final_accuracy:.3f if exp.final_accuracy else 'N/A'} | "
                f"{exp.best_eval_loss:.4f if exp.best_eval_loss else 'N/A'} | "
                f"{exp.precision} |"
            )

    lines.append("")
    lines.append("## Personalized Models")
    lines.append("")
    lines.append("| Experiment | Model | Dataset | CER↓ | WER↓ | NED↓ | Accuracy↑ | Loss↓ | Precision |")
    lines.append("|------------|-------|---------|------|------|------|-----------|-------|-----------|")

    for exp in experiments_sorted:
        if 'personalize' in exp.name.lower():
            lines.append(
                f"| {exp.name[:40]} | {exp.model} | {exp.dataset} | "
                f"{exp.final_cer:.3f if exp.final_cer else 'N/A'} | "
                f"{exp.final_wer:.3f if exp.final_wer else 'N/A'} | "
                f"{exp.final_ned:.3f if exp.final_ned else 'N/A'} | "
                f"{exp.final_accuracy:.3f if exp.final_accuracy else 'N/A'} | "
                f"{exp.best_eval_loss:.4f if exp.best_eval_loss else 'N/A'} | "
                f"{exp.precision} |"
            )

    lines.append("")
    lines.append("## Metrics Legend")
    lines.append("")
    lines.append("- **CER**: Character Error Rate (lower is better)")
    lines.append("- **WER**: Word Error Rate (lower is better)")
    lines.append("- **NED**: Normalized Edit Distance (lower is better)")
    lines.append("- **Accuracy**: Exact match accuracy (higher is better)")
    lines.append("- **Loss**: Best eval loss (lower is better)")
    lines.append("")
    lines.append(f"*Last updated: {Path.cwd()}*")

    return "\n".join(lines)


def generate_json_report(experiments: List[ExperimentMetrics], output_path: Path) -> None:
    """
    Generate JSON report with all experiment data.

    Args:
        experiments: List of ExperimentMetrics
        output_path: Output JSON file path
    """
    data = {
        'experiments': [exp.to_dict() for exp in experiments],
        'total_experiments': len(experiments)
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"JSON report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare training experiments and generate reports"
    )
    parser.add_argument(
        '--logs_dir',
        type=str,
        default='./ml/checkpoints/logs',
        help='Path to logs directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./ml/EXPERIMENT_COMPARISON.md',
        help='Output markdown file path'
    )
    parser.add_argument(
        '--json',
        type=str,
        help='Optional JSON output path'
    )
    parser.add_argument(
        '--experiments',
        nargs='+',
        help='Specific experiment names to compare (optional)'
    )

    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_path = Path(args.output)

    print("=" * 80)
    print("DisgraPhi Experiment Comparison")
    print("=" * 80)
    print(f"Logs directory: {logs_dir}")
    print(f"Output: {output_path}")
    print("=" * 80)

    # Load experiments
    if args.experiments:
        print(f"\nLoading {len(args.experiments)} specific experiments...")
        experiments = []
        for exp_name in args.experiments:
            exp_dir = logs_dir / exp_name
            metrics = load_tensorboard_metrics(exp_dir)
            if metrics:
                experiments.append(metrics)
    else:
        print(f"\nScanning {logs_dir} for experiments...")
        experiments = load_all_experiments(logs_dir)

    print(f"Found {len(experiments)} experiments")

    if not experiments:
        print("No experiments found. Exiting.")
        return

    # Generate markdown table
    print("\nGenerating comparison table...")
    table = generate_comparison_table(experiments)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(table)

    print(f"✓ Comparison table saved to: {output_path}")

    # Generate JSON report if requested
    if args.json:
        json_path = Path(args.json)
        generate_json_report(experiments, json_path)

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    if experiments:
        best_exp = min(experiments, key=lambda e: e.best_eval_loss if e.best_eval_loss else float('inf'))
        print(f"Best experiment: {best_exp.name}")
        print(f"  CER: {best_exp.final_cer:.3f}" if best_exp.final_cer else "  CER: N/A")
        print(f"  WER: {best_exp.final_wer:.3f}" if best_exp.final_wer else "  WER: N/A")
        print(f"  Loss: {best_exp.best_eval_loss:.4f}" if best_exp.best_eval_loss else "  Loss: N/A")

    print("=" * 80)


if __name__ == "__main__":
    main()
