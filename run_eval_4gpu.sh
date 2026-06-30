#!/bin/bash
# -----------------------------------------------------------------------------
# Drifting VLA (DBPO) - 4-GPU Evaluation Launcher Helper for pi05_drift_libero
# Launches parallel evaluations for pi05_drift_libero checkpoints on GPUs 0,1,2,3
# -----------------------------------------------------------------------------

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Make sure we are in the openpi/ directory
cd "$DIR"

# Run comparison evaluation
echo "Launching parallel evaluation on GPUs 0, 1, 2, 3..."
uv run scripts/run_eval_comparison.py \
    --checkpoint-base-dir checkpoints/pi05_drift_libero/my_drifting_vla_run \
    --checkpoint-config pi05_drift_libero \
    --gpus "0,1,2,3" \
    --num-trials 50

# Print final confirmation
echo "Evaluation complete! Reports are saved in checkpoints/pi05_drift_libero/my_drifting_vla_run/ or data/libero/checkpoint_comparison_report.md"
