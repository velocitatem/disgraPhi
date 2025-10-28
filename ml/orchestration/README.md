# GCP Vertex AI Orchestration

This module provides non-invasive integration with GCP Vertex AI for running DisgraPhi training jobs at scale. It allows you to:

- Run training jobs on powerful cloud GPUs
- Perform hyperparameter tuning with parallel trials
- Train multiple LoRA adapter configurations efficiently
- Maintain compatibility with local development workflows

## Architecture

The orchestration module is designed to be **non-invasive** - it hooks into the existing `ml/models/train.py` without modifying the core training code. This means:

- ✅ Training code remains unchanged
- ✅ Works locally without any cloud dependencies
- ✅ Can switch between local and cloud execution easily
- ✅ Testable without GCP credentials

### Components

```
ml/orchestration/
├── __init__.py              # Module exports
├── vertex_config.py         # Configuration classes
├── vertex_launcher.py       # Job submission and monitoring
├── vertex_cli.py           # Command-line interface
├── local_test.py           # Local testing utilities
├── examples.py             # Example configurations
└── README.md               # This file
```

## Prerequisites

### 1. Install Google Cloud SDK

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize gcloud
gcloud init

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Authenticate
gcloud auth login
gcloud auth application-default login
```

### 2. Enable Required APIs

```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Enable Cloud Storage API
gcloud services enable storage-api.googleapis.com
```

### 3. Create GCS Buckets

```bash
# Staging bucket for training packages
gsutil mb -l us-central1 gs://YOUR_PROJECT_ID-vertex-staging

# Data bucket (upload your IAM dataset)
gsutil mb -l us-central1 gs://YOUR_PROJECT_ID-data

# Checkpoints bucket
gsutil mb -l us-central1 gs://YOUR_PROJECT_ID-checkpoints
```

### 4. Upload IAM Dataset to GCS

```bash
# Upload your IAM dataset
gsutil -m cp -r ./ml/data/raw/iam gs://YOUR_PROJECT_ID-data/
```

## Quick Start

### Local Testing (No GCP Required)

Test your configuration locally before submitting:

```bash
# Test basic configuration
python -m ml.orchestration.local_test

# Test with your own config
python -c "
from ml.orchestration import VertexAIConfig
from ml.orchestration.local_test import LocalTester

config = VertexAIConfig(
    project_id='test-project',
    base_args={'model_provider': 'smolvlm-256m'}
)
tester = LocalTester(config)
tester.test_configuration()
"
```

### Single Training Job

Run a single training job with fixed hyperparameters:

```bash
python -m ml.orchestration.vertex_cli \
    --project-id YOUR_PROJECT_ID \
    --region us-central1 \
    --model-provider smolvlm-256m \
    --num-epochs 3 \
    --learning-rate 2e-5 \
    --batch-size 4 \
    --data-dir-gcs gs://YOUR_PROJECT_ID-data/iam \
    --output-dir-gcs gs://YOUR_PROJECT_ID-checkpoints
```

### Hyperparameter Tuning

Find the optimal LoRA configuration for bootstrap training:

```bash
python -m ml.orchestration.vertex_cli \
    --project-id YOUR_PROJECT_ID \
    --region us-central1 \
    --model-provider smolvlm-256m \
    --enable-tuning \
    --tune-lora-r 4,8,16,32 \
    --tune-lora-alpha 8,16,32,64 \
    --tune-learning-rate 1e-5,2e-5,5e-5 \
    --max-trials 20 \
    --parallel-trials 4 \
    --data-dir-gcs gs://YOUR_PROJECT_ID-data/iam \
    --output-dir-gcs gs://YOUR_PROJECT_ID-checkpoints
```

### Dry Run

Preview what will be submitted without actually submitting:

```bash
python -m ml.orchestration.vertex_cli \
    --project-id YOUR_PROJECT_ID \
    --model-provider smolvlm-256m \
    --enable-tuning \
    --tune-lora-r 4,8,16 \
    --dry-run
```

## Python API

For programmatic control, use the Python API:

```python
from ml.orchestration import (
    VertexAIConfig,
    VertexAILauncher,
    HyperparameterSpec,
    ParameterType
)

# Create configuration
config = VertexAIConfig(
    project_id="my-project",
    region="us-central1",
    enable_hyperparameter_tuning=True,
    hyperparameter_specs=[
        HyperparameterSpec(
            parameter_id="lora_r",
            parameter_type=ParameterType.DISCRETE,
            discrete_values=[4, 8, 16]
        ),
        HyperparameterSpec(
            parameter_id="learning_rate",
            parameter_type=ParameterType.DISCRETE,
            discrete_values=[1e-5, 2e-5, 5e-5]
        ),
    ],
    max_trial_count=9,
    parallel_trial_count=3,
    base_args={
        'model_provider': 'smolvlm-256m',
        'num_train_epochs': 3,
        'bf16': True,
    },
    data_dir_gcs="gs://my-project-data/iam",
    output_dir_gcs="gs://my-project-checkpoints",
)

# Submit job
launcher = VertexAILauncher(config)
job_id = launcher.submit_job()

# Monitor job
launcher.monitor_job(job_id)
```

## Use Cases

### 1. Bootstrap Training

Find the best LoRA configuration for the base IAM model:

```bash
python -m ml.orchestration.vertex_cli \
    --project-id YOUR_PROJECT_ID \
    --model-provider smolvlm-256m \
    --enable-tuning \
    --tune-lora-r 4,8,16,32 \
    --tune-lora-alpha 8,16,32 \
    --tune-learning-rate 1e-5,2e-5,5e-5 \
    --max-trials 30 \
    --parallel-trials 6 \
    --data-dir-gcs gs://YOUR_PROJECT_ID-data/iam \
    --output-dir-gcs gs://YOUR_PROJECT_ID-checkpoints/bootstrap
```

### 2. Model Comparison

Compare different model architectures:

```python
from ml.orchestration.examples import get_multi_model_comparison_config
from ml.orchestration import VertexAILauncher

config = get_multi_model_comparison_config("my-project")
launcher = VertexAILauncher(config)
job_id = launcher.submit_job()
```

### 3. Personalization

Fine-tune on user-specific data with a pre-trained bootstrap:

```bash
python -m ml.orchestration.vertex_cli \
    --project-id YOUR_PROJECT_ID \
    --model-provider smolvlm-256m \
    --dataset-type manifest \
    --manifest-data-dir gs://YOUR_PROJECT_ID-data/user123 \
    --bootstrap-adapter-path gs://YOUR_PROJECT_ID-checkpoints/bootstrap/best \
    --num-epochs 5 \
    --learning-rate 1e-5 \
    --output-dir-gcs gs://YOUR_PROJECT_ID-checkpoints/personalized
```

## Configuration Options

### Machine Types

Common machine types for training:

- `n1-standard-4`: 4 vCPUs, 15 GB RAM
- `n1-standard-8`: 8 vCPUs, 30 GB RAM (default)
- `n1-highmem-8`: 8 vCPUs, 52 GB RAM
- `n1-highmem-16`: 16 vCPUs, 104 GB RAM

### Accelerators

Available GPU types:

- `NVIDIA_TESLA_T4`: Entry-level, cost-effective (default)
- `NVIDIA_TESLA_V100`: High performance
- `NVIDIA_TESLA_P4`: Inference optimized
- `NVIDIA_TESLA_A100`: Cutting edge (limited availability)

### Hyperparameter Types

The module supports four types of hyperparameters:

1. **DISCRETE**: Fixed set of values (most common)
2. **CATEGORICAL**: String choices
3. **DOUBLE**: Continuous range
4. **INTEGER**: Integer range

## Cost Optimization

### Parallel Trials

Balance between speed and cost:

```bash
# Faster, more expensive (4 parallel trials)
--parallel-trials 4

# Slower, cheaper (2 parallel trials)
--parallel-trials 2
```

### Machine Size

Start small and scale up:

```bash
# Development/testing
--machine-type n1-standard-4 --accelerator-type NVIDIA_TESLA_T4

# Production
--machine-type n1-highmem-8 --accelerator-type NVIDIA_TESLA_V100
```

### Sample Ratio

Use smaller dataset samples for quick experiments:

```bash
# Quick test with 10% of data
--sample-ratio 0.1

# Full training
--sample-ratio 1.0
```

## Monitoring

### View Jobs in Console

After submission, you'll get a console link:

```
View in console: https://console.cloud.google.com/vertex-ai/training/custom-jobs/...
```

### CLI Monitoring

Monitor from the command line:

```bash
python -m ml.orchestration.vertex_cli \
    --project-id YOUR_PROJECT_ID \
    --monitor \
    ...other args...
```

### Check Logs

```bash
gcloud ai custom-jobs describe JOB_ID \
    --region us-central1 \
    --project YOUR_PROJECT_ID
```

## Troubleshooting

### Package Creation Fails

Ensure `setup.py` is properly configured:

```bash
python setup.py sdist
```

### GCS Upload Fails

Check bucket permissions:

```bash
gsutil ls gs://YOUR_PROJECT_ID-vertex-staging
```

### Job Submission Fails

Verify API is enabled:

```bash
gcloud services list --enabled | grep aiplatform
```

### Out of Memory

Try:
- Reduce batch size: `--batch-size 2`
- Use smaller model: `--model-provider smolvlm-256m`
- Increase machine size: `--machine-type n1-highmem-16`

## Advanced Usage

### Custom Container

Use a custom Docker container:

```python
config = VertexAIConfig(
    project_id="my-project",
    container_uri="gcr.io/my-project/custom-training:latest",
    ...
)
```

### Service Account

Run with a specific service account:

```bash
python -m ml.orchestration.vertex_cli \
    --service-account my-sa@my-project.iam.gserviceaccount.com \
    ...
```

### Network Configuration

Use a VPC network:

```bash
python -m ml.orchestration.vertex_cli \
    --network projects/PROJECT_NUM/global/networks/NETWORK \
    ...
```

## Examples

See `examples.py` for complete configuration examples:

```python
from ml.orchestration.examples import (
    get_basic_training_config,
    get_bootstrap_tuning_config,
    get_personalization_config,
    get_multi_model_comparison_config,
    get_large_scale_config,
)

# Get example config
config = get_bootstrap_tuning_config("my-project")

# Test locally
from ml.orchestration.local_test import LocalTester
tester = LocalTester(config)
tester.test_configuration()

# Submit to Vertex AI
from ml.orchestration import VertexAILauncher
launcher = VertexAILauncher(config)
job_id = launcher.submit_job()
```

## Best Practices

1. **Always test locally first**: Use `local_test.py` to validate configuration
2. **Start with dry runs**: Use `--dry-run` to see what will be submitted
3. **Use sample ratios for experiments**: Test with `--sample-ratio 0.1` first
4. **Monitor costs**: Check GCP billing dashboard regularly
5. **Clean up checkpoints**: Delete old checkpoints from GCS to save storage costs
6. **Use appropriate machine sizes**: Don't over-provision resources
7. **Leverage parallel trials**: Balance speed vs. cost based on urgency

## Integration with Existing Workflow

This module integrates seamlessly with your existing workflow:

```bash
# Local development (unchanged)
python ml/models/train.py --model-provider smolvlm-256m

# Cloud execution (new capability)
python -m ml.orchestration.vertex_cli \
    --project-id YOUR_PROJECT_ID \
    --model-provider smolvlm-256m

# Both use the same train.py code!
```

## Support

For issues or questions:
1. Check the examples in `examples.py`
2. Run local tests with `local_test.py`
3. Use `--dry-run` to preview job specs
4. Check GCP Vertex AI documentation: https://cloud.google.com/vertex-ai/docs

## License

Same as the main DisgraPhi project.
