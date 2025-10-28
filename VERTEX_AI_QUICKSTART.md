# GCP Vertex AI Training - Quick Start

This guide shows you how to run DisgraPhi training on GCP Vertex AI for hyperparameter tuning and large-scale training.

## What is this?

The `ml/orchestration/` module allows you to run your training jobs on powerful cloud GPUs without modifying the core training code. Perfect for:

- 🔬 Hyperparameter tuning (test multiple LoRA configs in parallel)
- 💪 Large-scale training (bigger GPUs, longer runs)
- 🚀 Batch experiments (compare models, datasets, etc.)

## Prerequisites

1. **GCP Project** with billing enabled
2. **gcloud CLI** installed ([guide](https://cloud.google.com/sdk/docs/install))
3. **Python dependencies**: Already in `requirements.txt`

## 5-Minute Setup

```bash
# 1. Configure GCP
gcloud init
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default login

# 2. Enable APIs
gcloud services enable aiplatform.googleapis.com storage-api.googleapis.com

# 3. Create buckets
gsutil mb -l us-central1 gs://YOUR_PROJECT_ID-vertex-staging
gsutil mb -l us-central1 gs://YOUR_PROJECT_ID-data
gsutil mb -l us-central1 gs://YOUR_PROJECT_ID-checkpoints

# 4. Upload IAM dataset
gsutil -m cp -r ./ml/data/raw/iam gs://YOUR_PROJECT_ID-data/
```

## Run Your First Job

### Test locally (no GCP needed)

```bash
python -m ml.orchestration.local_test
```

### Single training job

```bash
python -m ml.orchestration.vertex_cli \
    --project-id YOUR_PROJECT_ID \
    --model-provider smolvlm-256m \
    --num-epochs 3 \
    --data-dir-gcs gs://YOUR_PROJECT_ID-data/iam \
    --output-dir-gcs gs://YOUR_PROJECT_ID-checkpoints
```

### Hyperparameter tuning (THE MAIN USE CASE!)

```bash
python -m ml.orchestration.vertex_cli \
    --project-id YOUR_PROJECT_ID \
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

This will:
- Run 20 trials with different hyperparameter combinations
- Execute 4 trials in parallel (faster!)
- Find the best LoRA configuration for your model
- Save all checkpoints to GCS

## View Results

After submission, you'll get a console link:

```
✓ Job submitted successfully!
View in console: https://console.cloud.google.com/vertex-ai/training/...
```

Click the link to:
- Monitor training progress
- Compare trial results
- Download best checkpoints

## Full Documentation

See [`ml/orchestration/README.md`](ml/orchestration/README.md) for:
- Complete configuration options
- Cost optimization tips
- Advanced usage examples
- Troubleshooting guide

## Examples

Check out [`ml/orchestration/examples.py`](ml/orchestration/examples.py) for:
- Bootstrap training configurations
- Model comparison setups
- Personalization workflows
- Large-scale training configs

## Still want to run locally?

No problem! The original workflow is unchanged:

```bash
python ml/models/train.py --model-provider smolvlm-256m
```

The orchestration module is **completely optional** and non-invasive.

## Questions?

1. Read the [full documentation](ml/orchestration/README.md)
2. Check the [examples](ml/orchestration/examples.py)
3. Test locally with [`local_test.py`](ml/orchestration/local_test.py)
