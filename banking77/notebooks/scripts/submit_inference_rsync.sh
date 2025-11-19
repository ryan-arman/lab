#!/bin/bash

# Script to copy files to cluster using rsync and submit inference job
# More efficient for large files like test.jsonl
# Usage: ./submit_inference_rsync.sh [input_file] [config_file] [output_name] [adapter_path|cluster_host] [cluster_host]
# 
# Note: If 4th argument contains '@', it's treated as cluster_host (no adapter).
#       Otherwise, it's treated as adapter_path and 5th arg is cluster_host.
# 
# Examples:
#   ./submit_inference_rsync.sh                                    # Use defaults (base model)
#   ./submit_inference_rsync.sh data/test.jsonl                    # Custom input
#   ./submit_inference_rsync.sh data/test.jsonl configs/4b_instruct_vllm_infer.yaml results ryan@exun  # Base model, custom host
#   # With LoRA adapter (trained model):
#   ./submit_inference_rsync.sh data/test.jsonl configs/4b_instruct_vllm_infer.yaml results output/system_prompt_v2_lora_2757 ryan@exun

set -e

# Parse arguments
INPUT_FILE="${1:-data/banking77_test.jsonl}"
CONFIG_FILE="${2:-configs/4b_instruct_vllm_infer.yaml}"
OUTPUT_NAME="${3:-output}"

# Smart argument parsing: if 4th arg looks like a hostname (contains @), treat it as CLUSTER_HOST
# Otherwise, treat it as CHECKPOINT_PATH
if [ -n "${4}" ]; then
    if [[ "${4}" == *"@"* ]]; then
        # 4th argument is a hostname, so no checkpoint path provided
        CHECKPOINT_PATH=""
        CLUSTER_HOST="${4}"
    else
        # 4th argument is a checkpoint path
        CHECKPOINT_PATH="${4}"
        CLUSTER_HOST="${5:-ryan@exun}"
    fi
else
    # No 4th argument provided
    CHECKPOINT_PATH=""
    CLUSTER_HOST="${5:-ryan@exun}"
fi

# Configuration - adjust these for your cluster
CLUSTER_BASE_DIR="/home/ryan/code/oumi/lab/banking77/notebooks"
LOCAL_BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve paths (handle both relative and absolute paths)
if [[ "${INPUT_FILE}" != /* ]]; then
    # If path starts with .., resolve from current working directory
    if [[ "${INPUT_FILE}" == ../* ]]; then
        RESOLVED_DIR="$(cd "$(dirname "${INPUT_FILE}")" 2>/dev/null && pwd)"
        if [ -z "${RESOLVED_DIR}" ] || [ ! -d "${RESOLVED_DIR}" ]; then
            echo "Error: Could not resolve input file directory: $(dirname "${INPUT_FILE}")"
            exit 1
        fi
        INPUT_FILE="${RESOLVED_DIR}/$(basename "${INPUT_FILE}")"
    else
        INPUT_FILE="${LOCAL_BASE_DIR}/${INPUT_FILE}"
    fi
fi
if [[ "${CONFIG_FILE}" != /* ]]; then
    # If path starts with .., resolve from current working directory
    if [[ "${CONFIG_FILE}" == ../* ]]; then
        RESOLVED_DIR="$(cd "$(dirname "${CONFIG_FILE}")" 2>/dev/null && pwd)"
        if [ -z "${RESOLVED_DIR}" ] || [ ! -d "${RESOLVED_DIR}" ]; then
            echo "Error: Could not resolve config file directory: $(dirname "${CONFIG_FILE}")"
            exit 1
        fi
        CONFIG_FILE="${RESOLVED_DIR}/$(basename "${CONFIG_FILE}")"
    else
        CONFIG_FILE="${LOCAL_BASE_DIR}/${CONFIG_FILE}"
    fi
fi

# Extract just the filenames for cluster
INPUT_FILENAME=$(basename "${INPUT_FILE}")
CONFIG_FILENAME=$(basename "${CONFIG_FILE}")

echo "======================================"
echo "Copying files to cluster and submitting job (using rsync)"
echo "======================================"
echo "Cluster host: ${CLUSTER_HOST}"
echo "Cluster directory: ${CLUSTER_BASE_DIR}"
echo "Local directory: ${LOCAL_BASE_DIR}"
echo ""
echo "Input file: ${INPUT_FILE}"
echo "Config file: ${CONFIG_FILE}"
echo "Output name: ${OUTPUT_NAME}"
if [ -n "${CHECKPOINT_PATH}" ]; then
    echo "LoRA adapter path: ${CHECKPOINT_PATH}"
else
    echo "Adapter: (using base model)"
fi
echo ""

# Create remote directories if they don't exist
echo "Creating remote directories..."
ssh "${CLUSTER_HOST}" "mkdir -p ${CLUSTER_BASE_DIR} && mkdir -p ${CLUSTER_BASE_DIR}/logs"

# Copy files using rsync (more efficient, only copies changed files)
# Files are copied to flat structure on cluster for simplicity
echo ""
echo "Copying files to cluster using rsync..."

# Check if files exist
if [ ! -f "${INPUT_FILE}" ]; then
    echo "Error: Input file not found: ${INPUT_FILE}"
    exit 1
fi
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

rsync -avz --progress \
    "${INPUT_FILE}" \
    "${CONFIG_FILE}" \
    "${LOCAL_BASE_DIR}/scripts/run_inference.sh" \
    "${CLUSTER_HOST}:${CLUSTER_BASE_DIR}/"

echo ""
echo "Files copied successfully!"
echo ""

# Submit the job with environment variables
echo "Submitting SLURM job..."
# Build export string for sbatch
EXPORT_VARS="CONFIG_FILE=${CLUSTER_BASE_DIR}/${CONFIG_FILENAME},INPUT_PATH=${CLUSTER_BASE_DIR}/${INPUT_FILENAME},OUTPUT_NAME=${OUTPUT_NAME}"
if [ -n "${CHECKPOINT_PATH}" ]; then
    # Resolve checkpoint path (handle both relative and absolute)
    if [[ "${CHECKPOINT_PATH}" != /* ]]; then
        CHECKPOINT_PATH="${CLUSTER_BASE_DIR}/${CHECKPOINT_PATH}"
    fi
    EXPORT_VARS="${EXPORT_VARS},CHECKPOINT_PATH=${CHECKPOINT_PATH}"
fi

ssh "${CLUSTER_HOST}" "cd ${CLUSTER_BASE_DIR} && sbatch --export=${EXPORT_VARS} run_inference.sh"

echo ""
echo "======================================"
echo "Job submitted successfully!"
echo "======================================"
echo ""
echo "To check job status, run:"
echo "  ssh ${CLUSTER_HOST} 'squeue -u ryan'"
echo ""
echo "To view logs, run:"
echo "  ssh ${CLUSTER_HOST} 'tail -f ${CLUSTER_BASE_DIR}/logs/banking77_inference_qwen3_4b_*.log'"
echo ""

