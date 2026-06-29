#!/bin/bash
# -----------------------------------------------------------------------------
# Drifting VLA (DBPO) - 4-GPU Training Launcher Helper for pi05_drift_libero
# Launches training on GPUs 0, 1, 2, 3 using the parameterized run_train.sh
# -----------------------------------------------------------------------------

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Make run_train.sh executable just in case
chmod +x "$DIR/run_train.sh"

# Launch training using GPUs 0,1,2,3 and the pi05_drift_libero config
echo "Launching pi05_drift_libero training on GPUs 0, 1, 2, 3..."
"$DIR/run_train.sh" "0,1,2,3" "pi05_drift_libero"
