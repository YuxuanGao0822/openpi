#!/bin/bash
# -----------------------------------------------------------------------------
# DriftingVLA - Batch Ablation Run Script (8x A800 GPUs)
# Allows queuing multiple LCDF configs sequentially on all 8 GPUs.
# -----------------------------------------------------------------------------

# Usage:
#   # To run all Pi0 configurations:
#   ./run_all_experiments.sh pi0
#
#   # To run all Pi0.5 configurations:
#   ./run_all_experiments.sh pi05
#
#   # To run a specific set of configs:
#   ./run_all_experiments.sh pi0_drift_neg_libero pi0_drift_lcdf_full_libero

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
chmod +x "$DIR/run_train_8gpu.sh"

MODE=$1

if [ "$MODE" = "pi0" ]; then
    CONFIGS=(
        "pi0_libero"
        "pi0_drift_libero"
        "pi0_drift_neg_libero"
        "pi0_drift_lcdf_libero"
        "pi0_drift_lcdf_full_libero"
    )
elif [ "$MODE" = "pi05" ]; then
    CONFIGS=(
        "pi05_libero"
        "pi05_drift_libero"
        "pi05_drift_neg_libero"
        "pi05_drift_lcdf_libero"
        "pi05_drift_lcdf_full_libero"
    )
else
    # Treat arguments as a custom list of configs
    CONFIGS=("$@")
fi

echo "=================================================="
echo "Starting batch training of ${#CONFIGS[@]} configurations:"
for cfg in "${CONFIGS[@]}"; do
    echo "  - $cfg"
done
echo "=================================================="

for cfg in "${CONFIGS[@]}"; do
    echo ">>> Starting training for config: $cfg at $(date) <<<"
    
    # Run the 8-GPU script and WAIT for it to complete.
    # Note: run_train.sh inside uses nohup & in background, 
    # so we monitor the background PID to know when it finishes.
    
    "$DIR/run_train_8gpu.sh" "$cfg"
    
    # Wait for the train.py process of uv run to finish before moving to the next
    # Sleep to allow process launch
    sleep 10
    
    # Get PID of the running train.py
    PID=$(pgrep -f "train.py.*$cfg")
    
    if [ -n "$PID" ]; then
        echo "Monitoring training process (PID: $PID)..."
        while ps -p $PID > /dev/null; do
            sleep 30
        done
        echo ">>> Finished training for config: $cfg at $(date) <<<"
    else
        echo "Warning: Could not find running train.py process for $cfg. Moving to next config."
    fi
    echo "--------------------------------------------------"
done

echo "Batch training pipeline complete."
