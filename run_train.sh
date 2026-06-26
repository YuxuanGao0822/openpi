#!/bin/bash
# -----------------------------------------------------------------------------
# Drifting VLA (DBPO) - Training Launcher Script with Auto-Stats Computation
# Supports selectable single/multi-GPU execution.
# -----------------------------------------------------------------------------

# 1. Parse GPU configuration (defaults to GPU 0 if not provided)
# Usage: ./run_train.sh [gpu_ids] (e.g., ./run_train.sh 0,1)
GPUS=${1:-"0"}
FIRST_GPU=$(echo "$GPUS" | cut -d',' -f1)

# Count the number of GPUs
NUM_GPUS=$(echo "$GPUS" | tr -cd ',' | wc -c)
NUM_GPUS=$((NUM_GPUS + 1))

# Dynamically calculate global batch size to maintain a safe local batch size of 8 per GPU
BATCH_SIZE=$((NUM_GPUS * 8))

echo "Configured GPUs: $GPUS (Total: $NUM_GPUS, First: $FIRST_GPU, Global Batch Size: $BATCH_SIZE)"

# 2. Check and compute normalization statistics if missing
STATS_FILE="./assets/pi0_drift_libero/physical-intelligence/libero/norm_stats.json"
if [ ! -f "$STATS_FILE" ]; then
    echo "Normalization stats not found at $STATS_FILE."
    echo "Running compute_norm_stats.py on GPU $FIRST_GPU..."
    CUDA_VISIBLE_DEVICES=$FIRST_GPU uv run scripts/compute_norm_stats.py --config-name=pi0_drift_libero
else
    echo "Found normalization stats at $STATS_FILE. Skipping statistics computation."
fi

# 3. Restrict JAX and PyTorch visibility to configured GPUs
export CUDA_VISIBLE_DEVICES=$GPUS

# 4. Optimize JAX memory pre-allocation
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# 5. Launch training with nohup to run in the background
LOG_FILE="train.log"
echo "Starting Drifting VLA training in the background on GPU(s): $GPUS (fsdp-devices: $NUM_GPUS)..."
echo "Logs will be written to: $LOG_FILE"

nohup uv run scripts/train.py pi0_drift_libero \
    --exp-name my_drifting_vla_run \
    --overwrite \
    --batch-size $BATCH_SIZE \
    --fsdp-devices $NUM_GPUS > "$LOG_FILE" 2>&1 &

PID=$!
echo "Training launched in background with PID: $PID"
echo "--------------------------------------------------"
echo "To monitor the training progress, run:"
echo "  tail -f $LOG_FILE"
echo "--------------------------------------------------"



