#!/bin/bash

# Script to kill training jobs on the cluster
# Usage: ./kill_training.sh [job_id] [cluster_host]
# If job_id is not provided, kills the most recent training job

set -e

# Parse arguments
JOB_ID="$1"
CLUSTER_HOST="${2:-ryan@exun}"

echo "======================================"
echo "Killing training job on cluster"
echo "======================================"
echo "Cluster host: ${CLUSTER_HOST}"
echo ""

if [ -n "${JOB_ID}" ]; then
    echo "Killing job ${JOB_ID}..."
    ssh "${CLUSTER_HOST}" "scancel ${JOB_ID}"
    echo ""
    echo "Job ${JOB_ID} cancelled."
else
    echo "Finding most recent training job..."
    # Find the most recent training job for this user
    LATEST_JOB=$(ssh "${CLUSTER_HOST}" "squeue -u ryan -n banking77_training_qwen3_4b --format='%i' --noheader | head -1" || echo "")
    
    if [ -z "${LATEST_JOB}" ]; then
        echo "No running training jobs found."
        echo ""
        echo "To see all your jobs, run:"
        echo "  ssh ${CLUSTER_HOST} 'squeue -u ryan'"
        exit 0
    fi
    
    echo "Most recent training job: ${LATEST_JOB}"
    echo "Killing job ${LATEST_JOB}..."
    ssh "${CLUSTER_HOST}" "scancel ${LATEST_JOB}"
    echo ""
    echo "Job ${LATEST_JOB} cancelled."
fi

echo ""
echo "======================================"
echo "Job cancellation complete!"
echo "======================================"

