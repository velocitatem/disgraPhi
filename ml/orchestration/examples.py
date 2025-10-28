"""
Example configurations for Vertex AI training jobs.

These examples demonstrate different use cases for running training
on GCP Vertex AI.
"""

from ml.orchestration import VertexAIConfig, HyperparameterSpec, ParameterType


def get_basic_training_config(project_id: str) -> VertexAIConfig:
    """
    Basic single training job configuration.
    
    Use this for running a standard training job with fixed hyperparameters.
    """
    return VertexAIConfig(
        project_id=project_id,
        region="us-central1",
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        base_args={
            'model_provider': 'smolvlm-256m',
            'dataset_type': 'iam',
            'num_train_epochs': 3,
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 4,
            'learning_rate': 2e-5,
            'sample_ratio': 1.0,
            'bf16': True,
        },
        data_dir_gcs=f"gs://{project_id}-data/iam",
        output_dir_gcs=f"gs://{project_id}-checkpoints",
    )


def get_bootstrap_tuning_config(project_id: str) -> VertexAIConfig:
    """
    Hyperparameter tuning for bootstrap training on IAM.
    
    Use this to find the optimal LoRA configuration for the base model.
    """
    return VertexAIConfig(
        project_id=project_id,
        region="us-central1",
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        enable_hyperparameter_tuning=True,
        hyperparameter_specs=[
            HyperparameterSpec(
                parameter_id="lora_r",
                parameter_type=ParameterType.DISCRETE,
                discrete_values=[4, 8, 16, 32]
            ),
            HyperparameterSpec(
                parameter_id="lora_alpha",
                parameter_type=ParameterType.DISCRETE,
                discrete_values=[8, 16, 32, 64]
            ),
            HyperparameterSpec(
                parameter_id="learning_rate",
                parameter_type=ParameterType.DISCRETE,
                discrete_values=[1e-5, 2e-5, 5e-5]
            ),
        ],
        max_trial_count=20,
        parallel_trial_count=4,
        metric_id="eval_loss",
        metric_goal="MINIMIZE",
        base_args={
            'model_provider': 'smolvlm-256m',
            'dataset_type': 'iam',
            'num_train_epochs': 3,
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 4,
            'sample_ratio': 1.0,
            'bf16': True,
        },
        data_dir_gcs=f"gs://{project_id}-data/iam",
        output_dir_gcs=f"gs://{project_id}-checkpoints",
    )


def get_personalization_config(
    project_id: str,
    bootstrap_adapter_path: str,
    manifest_data_dir: str
) -> VertexAIConfig:
    """
    Configuration for personalization training.
    
    Use this for fine-tuning a pre-trained bootstrap model on user data.
    
    Args:
        project_id: GCP project ID
        bootstrap_adapter_path: Path to the bootstrap adapter (can be GCS path)
        manifest_data_dir: Path to user data manifest (can be GCS path)
    """
    return VertexAIConfig(
        project_id=project_id,
        region="us-central1",
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        base_args={
            'model_provider': 'smolvlm-256m',
            'dataset_type': 'manifest',
            'manifest_data_dir': manifest_data_dir,
            'bootstrap_adapter_path': bootstrap_adapter_path,
            'num_train_epochs': 5,
            'per_device_train_batch_size': 2,
            'gradient_accumulation_steps': 8,
            'learning_rate': 1e-5,
            'augment': True,
            'augment_strength': 0.7,
            'bf16': True,
        },
        output_dir_gcs=f"gs://{project_id}-checkpoints",
    )


def get_multi_model_comparison_config(project_id: str) -> VertexAIConfig:
    """
    Hyperparameter tuning across different models.
    
    Use this to compare different model architectures.
    """
    return VertexAIConfig(
        project_id=project_id,
        region="us-central1",
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        enable_hyperparameter_tuning=True,
        hyperparameter_specs=[
            HyperparameterSpec(
                parameter_id="model_provider",
                parameter_type=ParameterType.CATEGORICAL,
                categorical_values=[
                    "smolvlm-256m",
                    "qwen3-vl-2b",
                    "florence2-base"
                ]
            ),
            HyperparameterSpec(
                parameter_id="lora_r",
                parameter_type=ParameterType.DISCRETE,
                discrete_values=[8, 16]
            ),
        ],
        max_trial_count=6,
        parallel_trial_count=2,
        metric_id="eval_loss",
        metric_goal="MINIMIZE",
        base_args={
            'dataset_type': 'iam',
            'num_train_epochs': 3,
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 4,
            'learning_rate': 2e-5,
            'sample_ratio': 0.5,  # Use smaller sample for faster comparison
            'bf16': True,
        },
        data_dir_gcs=f"gs://{project_id}-data/iam",
        output_dir_gcs=f"gs://{project_id}-checkpoints",
    )


def get_large_scale_config(project_id: str) -> VertexAIConfig:
    """
    Configuration for large-scale training with powerful hardware.
    
    Use this for training larger models or full dataset training.
    """
    return VertexAIConfig(
        project_id=project_id,
        region="us-central1",
        machine_type="n1-highmem-16",
        accelerator_type="NVIDIA_TESLA_V100",
        accelerator_count=2,
        timeout_seconds=28800,  # 8 hours
        base_args={
            'model_provider': 'qwen3-vl-2b',
            'dataset_type': 'iam',
            'num_train_epochs': 10,
            'per_device_train_batch_size': 8,
            'gradient_accumulation_steps': 2,
            'learning_rate': 2e-5,
            'sample_ratio': 1.0,
            'bf16': True,
        },
        data_dir_gcs=f"gs://{project_id}-data/iam",
        output_dir_gcs=f"gs://{project_id}-checkpoints",
    )


# Example usage
if __name__ == '__main__':
    # Replace with your actual project ID
    PROJECT_ID = "my-gcp-project"
    
    # Example 1: Basic training
    print("Example 1: Basic Training")
    print("-" * 80)
    config = get_basic_training_config(PROJECT_ID)
    from ml.orchestration.local_test import LocalTester
    tester = LocalTester(config)
    tester.test_configuration()
    
    print("\n\nExample 2: Bootstrap Hyperparameter Tuning")
    print("-" * 80)
    config = get_bootstrap_tuning_config(PROJECT_ID)
    tester = LocalTester(config)
    tester.test_configuration()
    
    print(f"\nEstimated combinations: {tester.estimate_trial_combinations()}")
