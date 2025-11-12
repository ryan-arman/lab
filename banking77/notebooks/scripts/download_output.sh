#!/bin/bash

# Script to download inference output from cluster
# Usage: ./download_output.sh [job_id] [output_name] [cluster_host]
# If job_id is not provided, downloads the most recent output file
# If output_name is provided, searches for files matching <output_name>_*.jsonl

set -e

# Configuration
JOB_ID="$1"
OUTPUT_NAME="${2:-output}"
CLUSTER_HOST="${3:-ryan@exun}"
CLUSTER_BASE_DIR="/home/ryan/code/oumi/lab/banking77/notebooks"
CLUSTER_DATA_DIR="${CLUSTER_BASE_DIR}/data"
LOCAL_BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DATA_DIR="${LOCAL_BASE_DIR}/data"

echo "======================================"
echo "Downloading inference output from cluster"
echo "======================================"
echo "Cluster host: ${CLUSTER_HOST}"
echo "Cluster directory: ${CLUSTER_BASE_DIR}"
echo "Local directory: ${LOCAL_DATA_DIR}"
echo ""

# If job_id provided, download that specific file
if [ -n "${JOB_ID}" ]; then
    REMOTE_FILE="${CLUSTER_DATA_DIR}/${OUTPUT_NAME}_${JOB_ID}.jsonl"
    LOCAL_FILE="${LOCAL_DATA_DIR}/${OUTPUT_NAME}_${JOB_ID}.jsonl"
    
    echo "Downloading output for job ${JOB_ID} (output name: ${OUTPUT_NAME})..."
    echo "Remote: ${REMOTE_FILE}"
    echo "Local: ${LOCAL_FILE}"
    echo ""
    
    # Check if file exists on cluster
    if ssh "${CLUSTER_HOST}" "[ -f ${REMOTE_FILE} ]"; then
        scp "${CLUSTER_HOST}:${REMOTE_FILE}" "${LOCAL_FILE}"
        echo ""
        echo "File downloaded successfully!"
        echo "Size: $(ls -lh "${LOCAL_FILE}" | awk '{print $5}')"
        echo "Lines: $(wc -l < "${LOCAL_FILE}")"
    else
        echo "Error: Output file not found on cluster: ${REMOTE_FILE}"
        exit 1
    fi
else
    # Find the most recent output file on cluster matching the output name pattern
    echo "Finding most recent output file (pattern: ${OUTPUT_NAME}_*.jsonl)..."
    LATEST_FILE=$(ssh "${CLUSTER_HOST}" "ls -t ${CLUSTER_DATA_DIR}/${OUTPUT_NAME}_*.jsonl 2>/dev/null | head -1" || echo "")
    
    if [ -z "${LATEST_FILE}" ]; then
        echo "Error: No output files found matching pattern: ${OUTPUT_NAME}_*.jsonl in ${CLUSTER_DATA_DIR}"
        echo "Trying to find any output files in data directory..."
        LATEST_FILE=$(ssh "${CLUSTER_HOST}" "ls -t ${CLUSTER_DATA_DIR}/*_*.jsonl 2>/dev/null | head -1" || echo "")
        if [ -z "${LATEST_FILE}" ]; then
            echo "Error: No output files found on cluster"
            exit 1
        fi
    fi
    
    FILENAME=$(basename "${LATEST_FILE}")
    LOCAL_FILE="${LOCAL_DATA_DIR}/${FILENAME}"
    
    echo "Most recent file: ${LATEST_FILE}"
    echo "Downloading to: ${LOCAL_FILE}"
    echo ""
    
    scp "${CLUSTER_HOST}:${LATEST_FILE}" "${LOCAL_FILE}"
    
    echo ""
    echo "File downloaded successfully!"
    echo "File: ${FILENAME}"
    echo "Size: $(ls -lh "${LOCAL_FILE}" | awk '{print $5}')"
    echo "Lines: $(wc -l < "${LOCAL_FILE}")"
    
    # Extract job ID from filename (handles both output_123.jsonl and custom_name_123.jsonl)
    JOB_ID_FROM_FILE=$(echo "${FILENAME}" | sed 's/.*_\([0-9]*\)\.jsonl/\1/')
    echo "Job ID: ${JOB_ID_FROM_FILE}"
fi

echo ""
echo "======================================"
echo "Download complete!"
echo "======================================"

