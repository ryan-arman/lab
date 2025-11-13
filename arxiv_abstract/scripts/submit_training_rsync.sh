#!/bin/bash

# Script to copy files to cluster using rsync and submit training job
# More efficient for large files like training datasets
# Usage: ./submit_training_rsync.sh [train_dataset] [val_dataset] [config_file] [output_name] [cluster_host] [wandb_project] [wandb_entity]
# 
# Examples:
#   ./submit_training_rsync.sh                                    # Use defaults
#   ./submit_training_rsync.sh data/train.jsonl data/validation.jsonl  # Custom datasets
#   ./submit_training_rsync.sh "" "" "" "" "" my-custom-project  # Pass only wandb project (use "" for defaults)
#   ./submit_training_rsync.sh data/arxiv_summarization_train_instruct.jsonl data/arxiv_summarization_val_instruct.jsonl configs/qwen4b_train_lora.yaml my_training ryan@exun arxiv-abstract-qwen3-4b
#   ./submit_training_rsync.sh data/arxiv_summarization_train_instruct.jsonl data/arxiv_summarization_val_instruct.jsonl configs/qwen4b_train_lora.yaml my_training ryan@exun arxiv-abstract-qwen3-4b my-team
#
# Wandb Configuration:
#   - Set WANDB_API_KEY in your environment: export WANDB_API_KEY=your_key_here
#   - Or set it on the cluster: ssh ryan@exun 'echo "export WANDB_API_KEY=your_key" >> ~/.bashrc'
#   - Wandb project can be passed as argument 6, or set via WANDB_PROJECT environment variable
#   - Wandb entity (team) can be passed as argument 7, or set via WANDB_ENTITY environment variable
#   - IMPORTANT: If you get permission errors (403), the entity might not exist or you don't have access
#     Leave entity empty (pass "" as argument 7) to use your personal account instead

set -e

# Parse arguments (use empty string check to allow passing "" for defaults)
TRAIN_DATASET="${1:-data/arxiv_summarization_train_instruct.jsonl}"
VAL_DATASET="${2:-data/arxiv_summarization_val_instruct.jsonl}"
CONFIG_FILE="${3:-configs/qwen4b_train_lora.yaml}"
OUTPUT_NAME="${4:-arxiv_abstract_qwen3_4b_lora}"
CLUSTER_HOST="${5:-ryan@exun}"
# For wandb_project, check if argument 6 was explicitly provided (even if empty)
if [ $# -ge 6 ]; then
    WANDB_PROJECT="${6:-${WANDB_PROJECT:-arxiv-abstract}}"
else
    WANDB_PROJECT="${WANDB_PROJECT:-arxiv-abstract}"
fi
# For wandb_entity, check if argument 7 was explicitly provided
# If not provided, leave empty to use personal account (simpler, no permission issues)
if [ $# -ge 7 ]; then
    WANDB_ENTITY="${7:-${WANDB_ENTITY:-}}"
else
    WANDB_ENTITY="${WANDB_ENTITY:-}"
fi

# Configuration - adjust these for your cluster
CLUSTER_BASE_DIR="/home/ryan/code/oumi/lab/arxiv_abstract"
LOCAL_BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve paths (handle both relative and absolute paths)
if [[ "${TRAIN_DATASET}" != /* ]]; then
    TRAIN_DATASET="${LOCAL_BASE_DIR}/${TRAIN_DATASET}"
fi
if [[ "${VAL_DATASET}" != /* ]]; then
    VAL_DATASET="${LOCAL_BASE_DIR}/${VAL_DATASET}"
fi
if [[ "${CONFIG_FILE}" != /* ]]; then
    CONFIG_FILE="${LOCAL_BASE_DIR}/${CONFIG_FILE}"
fi

# Extract just the filenames for cluster
TRAIN_FILENAME=$(basename "${TRAIN_DATASET}")
VAL_FILENAME=$(basename "${VAL_DATASET}")
CONFIG_FILENAME=$(basename "${CONFIG_FILE}")

echo "======================================"
echo "Copying files to cluster and submitting training job (using rsync)"
echo "======================================"
echo "Cluster host: ${CLUSTER_HOST}"
echo "Cluster directory: ${CLUSTER_BASE_DIR}"
echo "Local directory: ${LOCAL_BASE_DIR}"
echo ""
echo "Train dataset: ${TRAIN_DATASET}"
echo "Validation dataset: ${VAL_DATASET}"
echo "Config file: ${CONFIG_FILE}"
echo "Output name: ${OUTPUT_NAME}"
echo "Wandb project: ${WANDB_PROJECT}"
if [ -n "${WANDB_ENTITY}" ]; then
    echo "Wandb entity: ${WANDB_ENTITY}"
    echo "⚠️  Note: Make sure you have access to entity '${WANDB_ENTITY}' or you'll get permission errors."
    echo "   If you get permission errors, leave entity empty to use your personal account."
else
    echo "Wandb entity: (using personal account)"
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "Wandb API key: Found in local environment (will be passed to cluster)"
else
    echo "Warning: WANDB_API_KEY not set in local environment."
    echo "Make sure it's set on the cluster or wandb may not work."
fi
echo ""

# Create remote directories if they don't exist
echo "Creating remote directories..."
ssh "${CLUSTER_HOST}" "mkdir -p ${CLUSTER_BASE_DIR} && mkdir -p ${CLUSTER_BASE_DIR}/logs && mkdir -p ${CLUSTER_BASE_DIR}/output"

# Copy files using rsync (more efficient, only copies changed files)
echo ""
echo "Copying files to cluster using rsync..."

# Check if files exist
if [ ! -f "${TRAIN_DATASET}" ]; then
    echo "Error: Training dataset not found: ${TRAIN_DATASET}"
    exit 1
fi
if [ ! -f "${VAL_DATASET}" ]; then
    echo "Error: Validation dataset not found: ${VAL_DATASET}"
    exit 1
fi
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

rsync -avz --progress \
    "${TRAIN_DATASET}" \
    "${VAL_DATASET}" \
    "${CONFIG_FILE}" \
    "${LOCAL_BASE_DIR}/scripts/run_training.sh" \
    "${CLUSTER_HOST}:${CLUSTER_BASE_DIR}/"

echo ""
echo "Files copied successfully!"
echo ""

# Submit the job with environment variables
echo "Submitting SLURM job..."
# Build export string for sbatch
EXPORT_VARS="CONFIG_FILE=${CLUSTER_BASE_DIR}/${CONFIG_FILENAME},TRAIN_DATASET=${CLUSTER_BASE_DIR}/${TRAIN_FILENAME},VAL_DATASET=${CLUSTER_BASE_DIR}/${VAL_FILENAME},OUTPUT_NAME=${OUTPUT_NAME},WANDB_PROJECT=${WANDB_PROJECT}"
if [ -n "${WANDB_ENTITY}" ]; then
    EXPORT_VARS="${EXPORT_VARS},WANDB_ENTITY=${WANDB_ENTITY}"
fi
# Pass WANDB_API_KEY from local environment to remote job
if [ -n "${WANDB_API_KEY:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},WANDB_API_KEY=${WANDB_API_KEY}"
    echo "Passing WANDB_API_KEY from local environment to cluster job"
fi

ssh "${CLUSTER_HOST}" "cd ${CLUSTER_BASE_DIR} && sbatch --export=${EXPORT_VARS} run_training.sh"

echo ""
echo "======================================"
echo "Job submitted successfully!"
echo "======================================"
echo ""
echo "To check job status, run:"
echo "  ssh ${CLUSTER_HOST} 'squeue -u ryan'"
echo ""
echo "To view logs, run:"
echo "  ssh ${CLUSTER_HOST} 'tail -f ${CLUSTER_BASE_DIR}/logs/arxiv_abstract_training_qwen3_4b_*.log'"
echo ""

