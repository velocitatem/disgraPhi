#!/bin/bash
# Batch Bootstrap Training Script for A100 (48GB VRAM)
# Trains multiple models in parallel and uploads to HuggingFace
#
# Prerequisites:
#   - HuggingFace CLI: pip install huggingface_hub
#   - Login: huggingface-cli login
#   - Set HF_USERNAME environment variable
#
# Usage:
#   HF_USERNAME=your-username ./ml/train_all_bootstrap.sh [sample_ratio]
#
# Example:
#   HF_USERNAME=myusername ./ml/train_all_bootstrap.sh 0.5  # 50% IAM dataset
#   HF_USERNAME=myusername ./ml/train_all_bootstrap.sh 1.0  # Full dataset (default)

set -e  # Exit on any error

# Configuration
SAMPLE_RATIO=${1:-1.0}
DATA_DIR="ml/data/raw/iam/processed"
OUTPUT_DIR="ml/checkpoints"
EPOCHS=3
EVAL_EVERY=250
SAVE_EVERY=500

# HuggingFace configuration
if [ -z "$HF_USERNAME" ]; then
    echo "ERROR: HF_USERNAME environment variable not set"
    echo "Usage: HF_USERNAME=your-username $0"
    exit 1
fi

HF_ORG="${HF_USERNAME}"  # Can change to organization name
HF_REPO_PREFIX="disgraphi-bootstrap"  # Prefix for repo names

# Logging
LOG_DIR="ml/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/batch_bootstrap_${TIMESTAMP}.log"

# GPU allocation for A100 48GB - can run 2 models in parallel
# SmolVLM-256M/500M: ~4-8GB each
# Qwen3-2B/4B: ~12-20GB each
PARALLEL_JOBS=2

echo "========================================" | tee "$MAIN_LOG"
echo "Batch Bootstrap Training - A100 Parallel Mode" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "Sample ratio: $SAMPLE_RATIO" | tee -a "$MAIN_LOG"
echo "Parallel jobs: $PARALLEL_JOBS" | tee -a "$MAIN_LOG"
echo "HuggingFace org: $HF_ORG" | tee -a "$MAIN_LOG"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" | tee -a "$MAIN_LOG"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)" | tee -a "$MAIN_LOG"
echo "Started: $(date)" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Check HuggingFace CLI is installed and logged in
if ! command -v huggingface-cli &> /dev/null; then
    echo "ERROR: huggingface-cli not found. Install with: pip install huggingface_hub" | tee -a "$MAIN_LOG"
    exit 1
fi

if ! huggingface-cli whoami &> /dev/null; then
    echo "ERROR: Not logged in to HuggingFace. Run: huggingface-cli login" | tee -a "$MAIN_LOG"
    exit 1
fi

echo "✓ HuggingFace CLI authenticated as: $(huggingface-cli whoami)" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Define models to train
# Format: "model_name quantization experiment_name vram_estimate_gb"
MODELS=(
    "smolvlm-256m 4bit smolvlm256m-4bit 6"
    "smolvlm-256m none smolvlm256m-fp16 8"
    "smolvlm-500m 4bit smolvlm500m-4bit 8"
    "qwen3-vl-2b 4bit qwen3-2b-4bit 14"
    "qwen3-vl-4b 4bit qwen3-4b-4bit 20"
)

# Function to upload model to HuggingFace
upload_to_huggingface() {
    local CHECKPOINT_DIR=$1
    local MODEL_NAME=$2
    local EXPERIMENT=$3

    echo "Uploading to HuggingFace: $CHECKPOINT_DIR" | tee -a "$MAIN_LOG"

    # Extract checkpoint type (best_model_hybrid, best_model_loss, final_model)
    CHECKPOINT_TYPE=$(basename "$CHECKPOINT_DIR")

    # Create repo name: disgraphi-bootstrap-smolvlm256m-4bit
    REPO_NAME="${HF_REPO_PREFIX}-${EXPERIMENT}"

    # Create README with model info
    cat > "$CHECKPOINT_DIR/README.md" <<EOF
---
license: apache-2.0
tags:
- handwriting-recognition
- vision-language
- lora
- disgraphi
- bootstrap
base_model: ${MODEL_NAME}
---

# DisgraPhi Bootstrap Adapter - ${EXPERIMENT}

Bootstrap LoRA adapter for handwriting recognition trained on IAM dataset.

## Model Details

- **Base Model**: ${MODEL_NAME}
- **Training Stage**: Bootstrap (global handwriting adapter)
- **Dataset**: IAM Handwriting Database (${SAMPLE_RATIO}x sampling)
- **Checkpoint**: ${CHECKPOINT_TYPE}
- **Architecture**: LoRA rank 8, alpha 16
- **Trained**: $(date)

## Usage

\`\`\`python
from ml.models.providers import create_model

# Load model with bootstrap adapter
model = create_model(
    model_type="${MODEL_NAME}",
    lora_r=8,
    lora_alpha=16,
    bootstrap_adapter_path="path/to/adapter"
)

# Generate transcription
text = model.generate(image, prompt="Transcribe this handwritten text.")
\`\`\`

## Personalization

This bootstrap adapter can be used as a frozen base for few-shot personalization:

\`\`\`bash
python train.py --mode personalize \\
  --model ${MODEL_NAME} \\
  --bootstrap path/to/this/adapter \\
  --user-dir data/user \\
  --user-rank 2 --augment --kl 0.5
\`\`\`

## Training Metrics

See TensorBoard logs for training curves.

## Citation

\`\`\`bibtex
@software{disgraphi2025,
  title={DisgraPhi: Hierarchical PEFT for Handwriting Recognition},
  author={Your Name},
  year={2025}
}
\`\`\`
EOF

    # Upload to HuggingFace
    python3 << PYEOF
from huggingface_hub import HfApi, create_repo
import os

api = HfApi()
repo_id = "${HF_ORG}/${REPO_NAME}"
checkpoint_dir = "${CHECKPOINT_DIR}"
checkpoint_type = "${CHECKPOINT_TYPE}"

try:
    # Create repo if it doesn't exist
    create_repo(repo_id, exist_ok=True, private=False)
    print(f"✓ Created/verified repo: {repo_id}")

    # Upload checkpoint folder
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        path_in_repo=checkpoint_type,
        commit_message=f"Upload {checkpoint_type} checkpoint"
    )
    print(f"✓ Uploaded {checkpoint_type} to {repo_id}")

except Exception as e:
    print(f"✗ Upload failed: {e}")
    exit(1)
PYEOF

    if [ $? -eq 0 ]; then
        echo "✓ Successfully uploaded to https://huggingface.co/${HF_ORG}/${REPO_NAME}" | tee -a "$MAIN_LOG"
        return 0
    else
        echo "✗ Upload failed for $CHECKPOINT_DIR" | tee -a "$MAIN_LOG"
        return 1
    fi
}

# Function to train a single model
train_model() {
    local MODEL=$1
    local QUANT=$2
    local EXPERIMENT=$3
    local VRAM=$4
    local GPU_ID=$5

    MODEL_LOG="$LOG_DIR/${EXPERIMENT}_${TIMESTAMP}.log"

    echo "[$EXPERIMENT] Starting on GPU $GPU_ID" | tee -a "$MAIN_LOG"

    # Build training command
    CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python ml/models/train.py \
        --mode bootstrap \
        --model $MODEL \
        --experiment-name $EXPERIMENT \
        --data-dir $DATA_DIR \
        --sample-ratio $SAMPLE_RATIO \
        --quant $QUANT \
        --epochs $EPOCHS \
        --eval-every $EVAL_EVERY \
        --save-every $SAVE_EVERY \
        --batch-size 2 \
        --grad-accum 4"

    echo "Command: $CMD" | tee -a "$MODEL_LOG"
    echo "Started: $(date)" | tee -a "$MODEL_LOG"

    # Run training with timeout
    TIMEOUT_HOURS=$( [[ $SAMPLE_RATIO == "1.0" ]] && echo 8 || echo 4 )
    TIMEOUT_SECONDS=$((TIMEOUT_HOURS * 3600))

    if timeout ${TIMEOUT_SECONDS}s bash -c "$CMD" >> "$MODEL_LOG" 2>&1; then
        echo "[$EXPERIMENT] Training completed successfully" | tee -a "$MAIN_LOG"

        # Upload all checkpoints to HuggingFace
        CHECKPOINT_BASE="$OUTPUT_DIR/bootstrap_${MODEL}/${EXPERIMENT}"

        if [ -d "$CHECKPOINT_BASE" ]; then
            echo "[$EXPERIMENT] Uploading checkpoints to HuggingFace..." | tee -a "$MAIN_LOG"

            # Upload each checkpoint type
            for CHECKPOINT_DIR in "$CHECKPOINT_BASE"/best_model_* "$CHECKPOINT_BASE"/final_model; do
                if [ -d "$CHECKPOINT_DIR" ]; then
                    upload_to_huggingface "$CHECKPOINT_DIR" "$MODEL" "$EXPERIMENT" || true
                fi
            done

            echo "[$EXPERIMENT] ✓ All checkpoints uploaded" | tee -a "$MAIN_LOG"
        else
            echo "[$EXPERIMENT] WARNING: Checkpoint directory not found: $CHECKPOINT_BASE" | tee -a "$MAIN_LOG"
        fi

        return 0
    else
        EXIT_CODE=$?
        echo "[$EXPERIMENT] ✗ Training failed with exit code: $EXIT_CODE" | tee -a "$MAIN_LOG"
        cp "$MODEL_LOG" "$LOG_DIR/FAILED_${EXPERIMENT}_${TIMESTAMP}.log"
        return 1
    fi
}

# Train models in parallel batches
TOTAL=${#MODELS[@]}
SUCCESS=0
FAILED=0
FAILED_MODELS=()

# Process models in batches based on VRAM requirements
# Simple strategy: pair small models together, run large models solo
i=0
while [ $i -lt $TOTAL ]; do
    IFS=' ' read -r MODEL1 QUANT1 EXPERIMENT1 VRAM1 <<< "${MODELS[$i]}"

    # Check if we can fit another model in parallel
    if [ $((i + 1)) -lt $TOTAL ] && [ $VRAM1 -le 10 ]; then
        IFS=' ' read -r MODEL2 QUANT2 EXPERIMENT2 VRAM2 <<< "${MODELS[$((i + 1))]}"

        # Only run in parallel if combined VRAM < 40GB (safe margin)
        if [ $((VRAM1 + VRAM2)) -lt 40 ]; then
            echo "" | tee -a "$MAIN_LOG"
            echo "========================================" | tee -a "$MAIN_LOG"
            echo "Training in parallel:" | tee -a "$MAIN_LOG"
            echo "  GPU 0: $MODEL1 ($QUANT1) - ${VRAM1}GB" | tee -a "$MAIN_LOG"
            echo "  GPU 0: $MODEL2 ($QUANT2) - ${VRAM2}GB (shared)" | tee -a "$MAIN_LOG"
            echo "========================================" | tee -a "$MAIN_LOG"

            # Train both in parallel on same GPU with MPS
            train_model "$MODEL1" "$QUANT1" "$EXPERIMENT1" "$VRAM1" 0 &
            PID1=$!

            sleep 10  # Stagger start to avoid initialization conflicts

            train_model "$MODEL2" "$QUANT2" "$EXPERIMENT2" "$VRAM2" 0 &
            PID2=$!

            # Wait for both to complete
            wait $PID1 && ((SUCCESS++)) || { ((FAILED++)); FAILED_MODELS+=("$MODEL1 ($QUANT1)"); }
            wait $PID2 && ((SUCCESS++)) || { ((FAILED++)); FAILED_MODELS+=("$MODEL2 ($QUANT2)"); }

            i=$((i + 2))
            continue
        fi
    fi

    # Run single model
    echo "" | tee -a "$MAIN_LOG"
    echo "========================================" | tee -a "$MAIN_LOG"
    echo "Training: $MODEL1 ($QUANT1) - ${VRAM1}GB" | tee -a "$MAIN_LOG"
    echo "========================================" | tee -a "$MAIN_LOG"

    train_model "$MODEL1" "$QUANT1" "$EXPERIMENT1" "$VRAM1" 0
    [ $? -eq 0 ] && ((SUCCESS++)) || { ((FAILED++)); FAILED_MODELS+=("$MODEL1 ($QUANT1)"); }

    i=$((i + 1))

    # Clear GPU cache between batches
    echo "Clearing GPU cache..." | tee -a "$MAIN_LOG"
    sleep 10
done

# Summary
echo "" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "Batch Training Complete" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "Total models: $TOTAL" | tee -a "$MAIN_LOG"
echo "Successful: $SUCCESS" | tee -a "$MAIN_LOG"
echo "Failed: $FAILED" | tee -a "$MAIN_LOG"

if [ $FAILED -gt 0 ]; then
    echo "" | tee -a "$MAIN_LOG"
    echo "Failed models:" | tee -a "$MAIN_LOG"
    for model in "${FAILED_MODELS[@]}"; do
        echo "  - $model" | tee -a "$MAIN_LOG"
    done
fi

echo "" | tee -a "$MAIN_LOG"
echo "All models uploaded to: https://huggingface.co/${HF_ORG}" | tee -a "$MAIN_LOG"
echo "Finished: $(date)" | tee -a "$MAIN_LOG"
echo "Main log: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"

# Exit with error if any model failed
[ $FAILED -eq 0 ] && exit 0 || exit 1
