#!/usr/bin/env python3
"""
Command-line interface for launching Vertex AI training jobs.

This script provides an easy way to submit training jobs to Vertex AI
for hyperparameter tuning and batch training without modifying the
core training code.

Examples:
    # Single training job
    python -m ml.orchestration.vertex_cli \
        --project-id my-project \
        --region us-central1 \
        --model-provider smolvlm-256m \
        --num-epochs 3 \
        --learning-rate 2e-5
    
    # Hyperparameter tuning
    python -m ml.orchestration.vertex_cli \
        --project-id my-project \
        --enable-tuning \
        --tune-lora-r 4,8,16 \
        --tune-learning-rate 1e-5,2e-5,5e-5 \
        --max-trials 10 \
        --parallel-trials 2
    
    # Dry run (see what would be submitted)
    python -m ml.orchestration.vertex_cli \
        --project-id my-project \
        --dry-run
"""

import argparse
import sys
from pathlib import Path
from typing import List

from ml.orchestration.vertex_config import (
    VertexAIConfig,
    HyperparameterSpec,
    ParameterType,
    ScaleType
)
from ml.orchestration.vertex_launcher import VertexAILauncher


def parse_list_arg(value: str, convert_fn=str) -> List:
    """Parse comma-separated list argument."""
    if not value:
        return []
    return [convert_fn(v.strip()) for v in value.split(',')]


def create_hyperparameter_specs(args) -> List[HyperparameterSpec]:
    """Create hyperparameter specifications from CLI arguments."""
    specs = []
    
    # LoRA rank tuning
    if args.tune_lora_r:
        values = parse_list_arg(args.tune_lora_r, int)
        specs.append(HyperparameterSpec(
            parameter_id="lora_r",
            parameter_type=ParameterType.DISCRETE,
            discrete_values=values
        ))
    
    # Learning rate tuning
    if args.tune_learning_rate:
        values = parse_list_arg(args.tune_learning_rate, float)
        specs.append(HyperparameterSpec(
            parameter_id="learning_rate",
            parameter_type=ParameterType.DISCRETE,
            discrete_values=values
        ))
    
    # Batch size tuning
    if args.tune_batch_size:
        values = parse_list_arg(args.tune_batch_size, int)
        specs.append(HyperparameterSpec(
            parameter_id="per_device_train_batch_size",
            parameter_type=ParameterType.DISCRETE,
            discrete_values=values
        ))
    
    # LoRA alpha tuning
    if args.tune_lora_alpha:
        values = parse_list_arg(args.tune_lora_alpha, int)
        specs.append(HyperparameterSpec(
            parameter_id="lora_alpha",
            parameter_type=ParameterType.DISCRETE,
            discrete_values=values
        ))
    
    # LoRA dropout tuning
    if args.tune_lora_dropout:
        values = parse_list_arg(args.tune_lora_dropout, float)
        specs.append(HyperparameterSpec(
            parameter_id="lora_dropout",
            parameter_type=ParameterType.DISCRETE,
            discrete_values=values
        ))
    
    # Warmup ratio tuning
    if args.tune_warmup_ratio:
        values = parse_list_arg(args.tune_warmup_ratio, float)
        specs.append(HyperparameterSpec(
            parameter_id="warmup_ratio",
            parameter_type=ParameterType.DISCRETE,
            discrete_values=values
        ))
    
    return specs


def main():
    parser = argparse.ArgumentParser(
        description="Launch DisgraPhi training on GCP Vertex AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # GCP Configuration
    gcp_group = parser.add_argument_group('GCP Configuration')
    gcp_group.add_argument(
        '--project-id', required=True,
        help='GCP project ID'
    )
    gcp_group.add_argument(
        '--region', default='us-central1',
        help='GCP region (default: us-central1)'
    )
    gcp_group.add_argument(
        '--staging-bucket',
        help='GCS bucket for staging (default: gs://{project-id}-vertex-staging)'
    )
    gcp_group.add_argument(
        '--service-account',
        help='Service account for Vertex AI jobs'
    )
    
    # Compute Configuration
    compute_group = parser.add_argument_group('Compute Configuration')
    compute_group.add_argument(
        '--machine-type', default='n1-standard-8',
        help='Machine type (default: n1-standard-8)'
    )
    compute_group.add_argument(
        '--accelerator-type', default='NVIDIA_TESLA_T4',
        help='Accelerator type (default: NVIDIA_TESLA_T4)'
    )
    compute_group.add_argument(
        '--accelerator-count', type=int, default=1,
        help='Number of accelerators (default: 1)'
    )
    compute_group.add_argument(
        '--timeout-hours', type=float, default=2.0,
        help='Job timeout in hours (default: 2.0)'
    )
    
    # Training Configuration (from train.py)
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument(
        '--model-provider', default='smolvlm-256m',
        help='Model provider (default: smolvlm-256m)'
    )
    train_group.add_argument(
        '--dataset-type', default='iam', choices=['iam', 'manifest'],
        help='Dataset type (default: iam)'
    )
    train_group.add_argument(
        '--data-dir-gcs',
        help='GCS path to dataset (e.g., gs://my-bucket/data/iam)'
    )
    train_group.add_argument(
        '--output-dir-gcs',
        help='GCS path for outputs (e.g., gs://my-bucket/checkpoints)'
    )
    train_group.add_argument(
        '--num-epochs', type=int, default=3,
        help='Number of training epochs (default: 3)'
    )
    train_group.add_argument(
        '--learning-rate', type=float, default=2e-5,
        help='Learning rate (default: 2e-5)'
    )
    train_group.add_argument(
        '--batch-size', type=int, default=4,
        help='Training batch size per device (default: 4)'
    )
    train_group.add_argument(
        '--gradient-accumulation-steps', type=int, default=4,
        help='Gradient accumulation steps (default: 4)'
    )
    train_group.add_argument(
        '--sample-ratio', type=float, default=1.0,
        help='Fraction of dataset to use (default: 1.0)'
    )
    train_group.add_argument(
        '--bootstrap-adapter-path',
        help='Path to bootstrap adapter for personalization'
    )
    train_group.add_argument(
        '--bf16', action='store_true',
        help='Use BF16 mixed precision'
    )
    train_group.add_argument(
        '--fp16', action='store_true',
        help='Use FP16 mixed precision'
    )
    
    # Hyperparameter Tuning
    tune_group = parser.add_argument_group('Hyperparameter Tuning')
    tune_group.add_argument(
        '--enable-tuning', action='store_true',
        help='Enable hyperparameter tuning'
    )
    tune_group.add_argument(
        '--max-trials', type=int, default=10,
        help='Maximum number of trials (default: 10)'
    )
    tune_group.add_argument(
        '--parallel-trials', type=int, default=2,
        help='Number of parallel trials (default: 2)'
    )
    tune_group.add_argument(
        '--metric-id', default='eval_loss',
        help='Metric to optimize (default: eval_loss)'
    )
    tune_group.add_argument(
        '--metric-goal', default='MINIMIZE', choices=['MINIMIZE', 'MAXIMIZE'],
        help='Optimization goal (default: MINIMIZE)'
    )
    
    # Hyperparameters to tune (comma-separated values)
    tune_group.add_argument(
        '--tune-lora-r',
        help='LoRA rank values to try (e.g., "4,8,16")'
    )
    tune_group.add_argument(
        '--tune-lora-alpha',
        help='LoRA alpha values to try (e.g., "8,16,32")'
    )
    tune_group.add_argument(
        '--tune-lora-dropout',
        help='LoRA dropout values to try (e.g., "0.05,0.1,0.15")'
    )
    tune_group.add_argument(
        '--tune-learning-rate',
        help='Learning rate values to try (e.g., "1e-5,2e-5,5e-5")'
    )
    tune_group.add_argument(
        '--tune-batch-size',
        help='Batch size values to try (e.g., "2,4,8")'
    )
    tune_group.add_argument(
        '--tune-warmup-ratio',
        help='Warmup ratio values to try (e.g., "0.05,0.1,0.15")'
    )
    
    # Job Control
    control_group = parser.add_argument_group('Job Control')
    control_group.add_argument(
        '--dry-run', action='store_true',
        help='Print job specification without submitting'
    )
    control_group.add_argument(
        '--monitor', action='store_true',
        help='Monitor job after submission'
    )
    control_group.add_argument(
        '--display-name',
        help='Custom display name for the job'
    )
    
    args = parser.parse_args()
    
    # Build base training arguments
    base_args = {
        'model_provider': args.model_provider,
        'dataset_type': args.dataset_type,
        'num_train_epochs': args.num_epochs,
        'per_device_train_batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate,
        'sample_ratio': args.sample_ratio,
    }
    
    # Add optional arguments
    if args.bootstrap_adapter_path:
        base_args['bootstrap_adapter_path'] = args.bootstrap_adapter_path
    if args.bf16:
        base_args['bf16'] = True
    if args.fp16:
        base_args['fp16'] = True
    
    # Create hyperparameter specs if tuning is enabled
    hyperparameter_specs = []
    if args.enable_tuning:
        hyperparameter_specs = create_hyperparameter_specs(args)
        if not hyperparameter_specs:
            print("Error: --enable-tuning requires at least one --tune-* parameter")
            sys.exit(1)
        
        print(f"Hyperparameter tuning enabled with {len(hyperparameter_specs)} parameters:")
        for spec in hyperparameter_specs:
            print(f"  - {spec.parameter_id}: {spec.discrete_values or spec.categorical_values}")
    
    # Create Vertex AI configuration
    config = VertexAIConfig(
        project_id=args.project_id,
        region=args.region,
        staging_bucket=args.staging_bucket,
        display_name=args.display_name,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        enable_hyperparameter_tuning=args.enable_tuning,
        hyperparameter_specs=hyperparameter_specs,
        max_trial_count=args.max_trials,
        parallel_trial_count=args.parallel_trials,
        metric_id=args.metric_id,
        metric_goal=args.metric_goal,
        base_args=base_args,
        data_dir_gcs=args.data_dir_gcs,
        output_dir_gcs=args.output_dir_gcs,
        service_account=args.service_account,
        timeout_seconds=int(args.timeout_hours * 3600),
    )
    
    # Create launcher and submit job
    try:
        launcher = VertexAILauncher(config)
        job_id = launcher.submit_job(dry_run=args.dry_run)
        
        if job_id and args.monitor:
            launcher.monitor_job(job_id)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
