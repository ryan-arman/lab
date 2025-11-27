#!/bin/bash

#SBATCH --job-name=banking77_training_qwen3_4b
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=128G
#SBATCH --output=/home/ryan/code/oumi/lab/banking77/notebooks/logs/banking77_training_qwen3_4b_%j.log
#SBATCH --error=/home/ryan/code/oumi/lab/banking77/notebooks/logs/banking77_training_qwen3_4b_%j.err
#SBATCH --time=48:00:00

# Example sbatch script for Banking77 training (supports both LoRA and full fine-tuning)
# Works with: configs/qwen4b_train_lora.yaml or configs/4b_instruct_full.yaml
#
# This script runs fine-tuning on the Banking77 dataset using Qwen3-4B-Instruct-2507

# Initialize conda and activate environment
# Try common conda initialization paths
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
fi

conda activate oumi

# Navigate to the working directory
cd /home/ryan/code/oumi

echo "======================================"
echo "Starting Banking77 Training"
echo "======================================"
echo "Job Name: banking77_training_qwen3_4b"
echo "GPUs: 8"
echo "Log: /home/ryan/code/oumi/lab/banking77/notebooks/logs/banking77_training_qwen3_4b_$SLURM_JOB_ID.log"
echo "======================================"

# Set paths (can be overridden via environment variables)
CONFIG_FILE="${CONFIG_FILE:-/home/ryan/code/oumi/lab/banking77/notebooks/configs/qwen4b_train_lora.yaml}"
TRAIN_DATASET="${TRAIN_DATASET:-/home/ryan/code/oumi/lab/banking77/notebooks/data/banking77_train.jsonl}"
VAL_DATASET="${VAL_DATASET:-/home/ryan/code/oumi/lab/banking77/notebooks/data/banking77_val.jsonl}"
TRAIN_DATASET_2="${TRAIN_DATASET_2:-}"  # Optional: second training dataset for mixing
OUTPUT_NAME="${OUTPUT_NAME:-banking77_qwen3_4b_lora}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/ryan/code/oumi/lab/banking77/notebooks/output/${OUTPUT_NAME}_${SLURM_JOB_ID}}"
RUN_NAME="${RUN_NAME:-${OUTPUT_NAME}_${SLURM_JOB_ID}}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"  # Optional: path to checkpoint to resume from

# Wandb configuration (can be overridden via environment variables)
WANDB_PROJECT="${WANDB_PROJECT:-banking77}"
WANDB_ENTITY="${WANDB_ENTITY:-}"  # Leave empty to use personal account (your username), or set to team name
# WANDB_API_KEY should be set in your environment (e.g., in ~/.bashrc on cluster)
# Try to source it from common locations if not already set
if [ -z "${WANDB_API_KEY:-}" ]; then
    # Try to source from bashrc if it exists
    if [ -f "$HOME/.bashrc" ]; then
        source "$HOME/.bashrc" 2>/dev/null || true
    fi
    # Also try .bash_profile
    if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.bash_profile" ]; then
        source "$HOME/.bash_profile" 2>/dev/null || true
    fi
    # Also try .profile
    if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.profile" ]; then
        source "$HOME/.profile" 2>/dev/null || true
    fi
fi

echo "Config file: ${CONFIG_FILE}"
echo "Train dataset: ${TRAIN_DATASET}"
if [ -n "${TRAIN_DATASET_2}" ]; then
    echo "Second train dataset: ${TRAIN_DATASET_2}"
fi
echo "Validation dataset: ${VAL_DATASET}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Run name: ${RUN_NAME}"
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    echo "Resume from checkpoint: ${RESUME_FROM_CHECKPOINT}"
else
    echo "Resume from checkpoint: (none - starting from scratch)"
fi
echo "Wandb project: ${WANDB_PROJECT}"
echo "Wandb run name: ${RUN_NAME}"
if [ -n "${WANDB_ENTITY}" ]; then
    echo "Wandb entity: ${WANDB_ENTITY}"
else
    echo "Wandb entity: (using personal account)"
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "Wandb API key: Set (${#WANDB_API_KEY} characters)"
else
    echo "Wandb API key: Not set - ERROR!"
fi
echo ""

# Print full config file for reference
echo "======================================"
echo "Full Configuration File:"
echo "======================================"
cat "${CONFIG_FILE}"
echo ""
echo "======================================"
echo ""

# Check if datasets exist
if [ ! -f "${TRAIN_DATASET}" ]; then
    echo "Error: Training dataset not found: ${TRAIN_DATASET}"
    exit 1
fi
if [ ! -f "${VAL_DATASET}" ]; then
    echo "Error: Validation dataset not found: ${VAL_DATASET}"
    exit 1
fi

echo "Running training..."
echo ""

# Set wandb environment variables
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_RUN_NAME="${RUN_NAME}"  # Explicitly set wandb run name
# Note: If you get permission errors, the entity might not exist or you don't have access
# Leave WANDB_ENTITY empty to use your personal account instead
if [ -n "${WANDB_ENTITY}" ]; then
    export WANDB_ENTITY="${WANDB_ENTITY}"
    echo "Using wandb entity: ${WANDB_ENTITY}"
    echo "If you get permission errors, try leaving entity empty to use personal account."
else
    # If entity not set, unset it to use personal account
    unset WANDB_ENTITY
    echo "Using personal wandb account (no entity specified)"
fi
# WANDB_API_KEY should be passed from submit script via sbatch --export
# If not set, try to source it from common locations
if [ -z "${WANDB_API_KEY:-}" ]; then
    # Try to source from bashrc if it exists
    if [ -f "$HOME/.bashrc" ]; then
        source "$HOME/.bashrc" 2>/dev/null || true
    fi
    # Also try .bash_profile
    if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.bash_profile" ]; then
        source "$HOME/.bash_profile" 2>/dev/null || true
    fi
    # Also try .profile
    if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.profile" ]; then
        source "$HOME/.profile" 2>/dev/null || true
    fi
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY="${WANDB_API_KEY}"
else
    echo "Error: WANDB_API_KEY is required but not set!"
    echo "Please set it in your local environment before running submit_training_rsync.sh,"
    echo "or set it in ~/.bashrc on the cluster."
    echo "You can get your API key from: https://wandb.ai/authorize"
    exit 1
fi

# Run training with command-line overrides for dataset paths and output
# Build the command with optional resume_from_checkpoint and second dataset parameters
TRAIN_CMD="oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
    -c \"${CONFIG_FILE}\" \
    --data.train.datasets.0.dataset_path=\"${TRAIN_DATASET}\" \
    --data.validation.datasets.0.dataset_path=\"${VAL_DATASET}\" \
    --training.output_dir=\"${OUTPUT_DIR}\" \
    --training.run_name=\"${RUN_NAME}\""

# Add second training dataset if provided
if [ -n "${TRAIN_DATASET_2}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --data.train.datasets.1.dataset_path=\"${TRAIN_DATASET_2}\""
    echo "Using two training datasets with mixture strategy"
fi

# Add resume_from_checkpoint if provided
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --training.resume_from_checkpoint=\"${RESUME_FROM_CHECKPOINT}\""
    echo "Resuming training from checkpoint: ${RESUME_FROM_CHECKPOINT}"
fi

# Execute the training command
eval ${TRAIN_CMD}

echo ""
echo "======================================"
echo "Training complete!"
echo "Job ID: $SLURM_JOB_ID"
echo "Output directory: ${OUTPUT_DIR}"
echo "======================================"

