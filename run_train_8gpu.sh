#!/bin/bash
# -----------------------------------------------------------------------------
# Drifting VLA (DBPO) - 8-GPU Training Launcher Helper
# Launches training on all 8 GPUs (0-7) using the parameterized run_train.sh
# -----------------------------------------------------------------------------

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Make run_train.sh executable
chmod +x "$DIR/run_train.sh"

CONFIG_NAME=${1:-"pi0_drift_lcdf_full_libero"}

echo "Launching $CONFIG_NAME training on all 8 GPUs (0,1,2,3,4,5,6,7)..."
"$DIR/run_train.sh" "0,1,2,3,4,5,6,7" "$CONFIG_NAME"
