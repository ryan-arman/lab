#!/bin/bash

#SBATCH --job-name=arxiv_abstract_inference_qwen3_4b
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=128G
#SBATCH --output=/home/ryan/code/oumi/lab/arxiv_abstract/logs/arxiv_abstract_inference_qwen3_4b_%j.log
#SBATCH --error=/home/ryan/code/oumi/lab/arxiv_abstract/logs/arxiv_abstract_inference_qwen3_4b_%j.err
#SBATCH --time=04:00:00

# Example sbatch script for Arxiv Abstract Summarization inference
# Generated from: configs/4b_instruct_vllm_infer.yaml
#
# This script runs inference on the Arxiv Abstract Summarization dataset using Qwen3-4B-Instruct-2507

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
echo "Starting Arxiv Abstract Inference"
echo "======================================"
echo "Job Name: arxiv_abstract_inference_qwen3_4b"
echo "GPUs: 1"
echo "Log: /home/ryan/code/oumi/lab/arxiv_abstract/logs/arxiv_abstract_inference_qwen3_4b_$SLURM_JOB_ID.log"
echo "======================================"

# Set paths (adjust these to match your cluster's file system)
# These can be overridden via environment variables when submitting:
#   sbatch --export=CONFIG_FILE=...,INPUT_PATH=...,OUTPUT_NAME=...,CHECKPOINT_PATH=... run_inference.sh
CONFIG_FILE="${CONFIG_FILE:-/home/ryan/code/oumi/lab/arxiv_abstract/configs/4b_instruct_vllm_infer.yaml}"
INPUT_PATH="${INPUT_PATH:-/home/ryan/code/oumi/lab/arxiv_abstract/data/arxiv_summarization_test_instruct.jsonl}"
OUTPUT_NAME="${OUTPUT_NAME:-output}"
OUTPUT_PATH="${OUTPUT_PATH:-/home/ryan/code/oumi/lab/arxiv_abstract/data/${OUTPUT_NAME}_${SLURM_JOB_ID}.jsonl}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"  # Optional: path to LoRA adapter checkpoint directory

echo "Config file: ${CONFIG_FILE}"
echo "Input path: ${INPUT_PATH}"
echo "Output path: ${OUTPUT_PATH}"
if [ -n "${CHECKPOINT_PATH}" ]; then
    echo "LoRA adapter path: ${CHECKPOINT_PATH}"
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

# Check if input file exists (for batch mode)
if [ -f "${INPUT_PATH}" ]; then
    echo "Running batch inference..."
    echo "Input file: ${INPUT_PATH}"
    echo "Output file: ${OUTPUT_PATH}"
    echo ""
    
    # Run batch inference
    # If checkpoint (adapter path) is provided, load base model with LoRA adapter
    if [ -n "${CHECKPOINT_PATH}" ]; then
        # Check if adapter_config.json exists in the provided path
        # If not, try checking in output/ subdirectory, then look for checkpoint subdirectories
        ADAPTER_PATH=""
        CLUSTER_BASE_DIR="/home/ryan/code/oumi/lab/arxiv_abstract"
        
        # Function to find adapter path
        find_adapter_path() {
            local test_path="$1"
            # Check if adapter_config.json exists directly in the path
            if [ -f "${test_path}/adapter_config.json" ]; then
                echo "${test_path}"
                return 0
            fi
            # Check in checkpoint subdirectories
            if [ -d "${test_path}" ]; then
                local latest_checkpoint=$(find "${test_path}" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -n 1)
                if [ -n "${latest_checkpoint}" ] && [ -f "${latest_checkpoint}/adapter_config.json" ]; then
                    echo "${latest_checkpoint}"
                    return 0
                fi
            fi
            return 1
        }
        
        # Try the provided path first
        if ADAPTER_PATH=$(find_adapter_path "${CHECKPOINT_PATH}" 2>/dev/null); then
            if [ "${ADAPTER_PATH}" != "${CHECKPOINT_PATH}" ]; then
                echo "Found adapter in checkpoint subdirectory: ${ADAPTER_PATH}"
            fi
        else
            # Try in output/ subdirectory (training saves to output/NAME_JOBID)
            # Extract just the directory name from the path
            CHECKPOINT_DIRNAME=$(basename "${CHECKPOINT_PATH}")
            OUTPUT_SUBDIR="${CLUSTER_BASE_DIR}/output/${CHECKPOINT_DIRNAME}"
            if [ -d "${OUTPUT_SUBDIR}" ]; then
                echo "adapter_config.json not found in ${CHECKPOINT_PATH}, checking output subdirectory: ${OUTPUT_SUBDIR}"
                if ADAPTER_PATH=$(find_adapter_path "${OUTPUT_SUBDIR}" 2>/dev/null); then
                    echo "Found adapter in: ${ADAPTER_PATH}"
                else
                    echo "Error: Could not find adapter_config.json in:"
                    echo "  - ${CHECKPOINT_PATH}"
                    echo "  - ${OUTPUT_SUBDIR}"
                    echo "  - or any checkpoint subdirectories"
                    exit 1
                fi
            else
                echo "Error: Could not find adapter_config.json in ${CHECKPOINT_PATH} or any checkpoint subdirectories"
                echo "Also checked: ${OUTPUT_SUBDIR} (directory not found)"
                exit 1
            fi
        fi
        echo "Loading base model with LoRA adapter from: ${ADAPTER_PATH}"
        oumi infer \
            -c "${CONFIG_FILE}" \
            --model.adapter_model="${ADAPTER_PATH}" \
            --input_path "${INPUT_PATH}" \
            --output_path "${OUTPUT_PATH}"
    else
        echo "Using base model (no adapter specified)"
        oumi infer \
            -c "${CONFIG_FILE}" \
            --input_path "${INPUT_PATH}" \
            --output_path "${OUTPUT_PATH}"
    fi
else
    echo "Input file not found: ${INPUT_PATH}"
    echo "Running interactive inference mode..."
    echo ""
    
    # Run interactive inference
    oumi infer \
        -i \
        -c "${CONFIG_FILE}"
fi

echo ""
echo "======================================"
echo "Inference complete!"
echo "Job ID: $SLURM_JOB_ID"
if [ -f "${OUTPUT_PATH}" ]; then
    echo "Output saved to: ${OUTPUT_PATH}"
fi
echo "======================================"

